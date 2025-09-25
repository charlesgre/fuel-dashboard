# litasco_runs.py
from __future__ import annotations

import platform
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================
# Réglages / chemins
# =========================
# 1) Dossier réseau (UNC)
RUNS_DIR = r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\Litasco balances\Litasco supply"

# 2) Dossier local du repo (fallback si l'UNC n'est pas monté)
REPO_RUNS_DIR = Path(__file__).resolve().parent / "Litasco balances" / "Litasco supply"

# 3) Fichier joint local (dernier recours)
FALLBACK_FILE = Path("/mnt/data/Europe Runs Recap 09.16.2025.xlsx")

# Motif des fichiers (accepte .xlsx / .xls)
FILENAME_PREFIX = "Europe Runs Recap"
FILENAME_GLOB = f"{FILENAME_PREFIX} *.xls*"


# =========================
# Utilitaires fichiers
# =========================
def _is_windows() -> bool:
    return platform.system() == "Windows"


def _latest_in(folder: Path) -> Path | None:
    try:
        if folder.exists():
            files = sorted(
                folder.glob(FILENAME_GLOB),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if files:
                return files[0]
    except Exception:
        return None
    return None


def pick_latest_runs_file() -> Path:
    """
    Ordre de recherche :
      1) UNC (RUNS_DIR)
      2) Dossier local du repo (REPO_RUNS_DIR)
      3) Fichier joint (FALLBACK_FILE)
    """
    p = _latest_in(Path(RUNS_DIR))
    if p:
        return p

    p = _latest_in(REPO_RUNS_DIR)
    if p:
        return p

    if FALLBACK_FILE.exists():
        return FALLBACK_FILE

    raise FileNotFoundError(
        f"Aucun fichier {FILENAME_GLOB} trouvé dans:\n"
        f"- UNC: {RUNS_DIR}\n- Repo: {REPO_RUNS_DIR}\n"
        f"et fallback absent: {FALLBACK_FILE}"
    )


# =========================
# Chargement & préparation
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def load_runs_workbook(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Lit le classeur et renvoie {sheet_name: DataFrame}, après nettoyage (lignes/colonnes vides).
    Supporte .xlsx (openpyxl) et .xls (xlrd pour .xls uniquement).
    """
    suffix = xlsx_path.suffix.lower()
    engine = None
    if suffix == ".xlsx":
        engine = "openpyxl"
    elif suffix == ".xls":
        engine = "xlrd"  # nécessite xlrd compatible .xls

    try:
        wb = pd.read_excel(xlsx_path, sheet_name=None, engine=engine)
    except Exception as e:
        if suffix == ".xls":
            raise RuntimeError(
                "Lecture .xls impossible. Installe 'xlrd' compatible .xls ou convertis en .xlsx."
            ) from e
        raise

    cleaned: Dict[str, pd.DataFrame] = {}
    for sh, df in wb.items():
        cleaned[sh] = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return cleaned


def _pick_region_sheet(wb: Dict[str, pd.DataFrame], region: str) -> Tuple[str, pd.DataFrame]:
    """
    region ∈ {'NWE','MED'}.
    Si des feuilles 'NWE' / 'MED' existent -> on les prend.
    Sinon: 1ère feuille = NWE, 2ème = MED.
    """
    keys = list(wb.keys())
    if not keys:
        raise ValueError("Classeur vide : aucune feuille détectée.")
    lower = {k.lower(): k for k in keys}
    if region.upper() == "NWE":
        sh = lower.get("nwe", keys[0])
    else:
        sh = lower.get("med", keys[1] if len(keys) > 1 else keys[0])
    return sh, wb[sh]


def tidy_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entrée wide:
      - 1ère colonne = dates -> 'Date'
      - colonnes suivantes = pays
    Sortie long:
      [Date, Year, Month, MonthLabel, Country, Value]
    """
    if df.empty:
        return pd.DataFrame(columns=["Date", "Year", "Month", "MonthLabel", "Country", "Value"])

    df = df.copy()
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Date"})

    # Parsing dates robuste
    def parse_date(x):
        if pd.isna(x):
            return np.nan
        try:
            return pd.to_datetime(x, errors="coerce", infer_datetime_format=True)
        except Exception:
            return pd.NaT

    df["Date"] = df["Date"].map(parse_date)
    df = df[df["Date"].notna()].copy()

    value_cols = [c for c in df.columns if c != "Date"]
    long_df = df.melt(id_vars="Date", value_vars=value_cols, var_name="Country", value_name="Value")

    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Value"])

    long_df["Year"] = long_df["Date"].dt.year.astype(int)
    long_df["Month"] = long_df["Date"].dt.month.astype(int)
    long_df["MonthLabel"] = pd.Categorical(
        long_df["Date"].dt.strftime("%b"),
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ordered=True,
    )

    return long_df.sort_values(["Country", "Year", "Month"]).reset_index(drop=True)


# =========================
# Tracés
# =========================
def plot_country_seasonal(long_df: pd.DataFrame, country: str, unit: str | None = None) -> go.Figure:
    """
    Seasonal Jan→Déc, une ligne par année pour un pays donné.
    """
    d = long_df[long_df["Country"] == country]
    fig = go.Figure()

    if d.empty:
        fig.update_layout(title=f"{country} — (aucune donnée)", template="plotly_white", height=300)
        return fig

    pivot = d.pivot_table(index="Month", columns="Year", values="Value",
                          aggfunc="mean").sort_index()

    for year in pivot.columns:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, 13)),
                y=pivot[year].reindex(range(1, 13)).values,
                mode="lines+markers",
                name=str(year),
                connectgaps=True,
            )
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(title_text=(unit or ""), rangemode="tozero")
    fig.update_layout(
        title=country,
        legend_title_text="Année",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_region_seasonals(long_df: pd.DataFrame, countries: List[str], unit: str | None) -> Dict[str, go.Figure]:
    return {c: plot_country_seasonal(long_df, c, unit=unit) for c in countries}


# =========================
# UI Streamlit
# =========================
def run_litasco_runs_tab():
    st.subheader("Litasco runs – Seasonals par pays")

    # Sélecteur de région
    region = st.radio("Région", ["NWE", "MED"], horizontal=True)

    # Choix automatique + override manuel
    with st.expander("📁 Fichier utilisé (auto: plus récent)"):
        st.caption(f"Dossier UNC : {RUNS_DIR}")
        st.caption(f"Dossier local du repo : {REPO_RUNS_DIR}")
        if not _is_windows() and (RUNS_DIR.startswith("\\") or RUNS_DIR.startswith("//")):
            st.warning(
                "Chemin UNC détecté sur un runtime non-Windows : peut ne pas être accessible.\n"
                "➡️ Le code tentera d'utiliser le repo local, puis la pièce jointe."
            )
        override = st.text_input(
            "Override (chemin/nom de fichier .xls/.xlsx)",
            value="",
            help="Laisse vide pour auto-pick"
        )
        try:
            xlsx_path = Path(override) if override.strip() else pick_latest_runs_file()
            st.info(f"Fichier choisi : **{xlsx_path.name}**")
        except Exception as e:
            st.error(str(e))
            st.stop()

    # Chargement du classeur
    try:
        wb = load_runs_workbook(xlsx_path)
    except Exception as e:
        st.exception(e)
        st.stop()

    # Feuille selon région
    sheet_name, df_region = _pick_region_sheet(wb, region)
    st.caption(f"Feuille utilisée pour {region} : **{sheet_name}**")

    # Mise en forme
    long_df = tidy_runs_df(df_region)

    # Unité (optionnelle)
    unit = st.text_input("Unité (axe Y)", value="", help="ex: kb/d (optionnel)")

    # Pays à afficher
    all_countries = sorted(long_df["Country"].unique().tolist())
    selected = st.multiselect("Pays à afficher", all_countries, default=all_countries)

    # Affichage en grille 3 colonnes
    cols = st.columns(3)
    for i, country in enumerate(selected):
        fig = plot_country_seasonal(long_df, country, unit=unit)
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True, key=f"runs_{region}_{country}_{i}")
