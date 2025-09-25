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
# Dossier réseau où sont stockés les "Europe Runs Recap *.xls*"
RUNS_DIR = r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\Litasco balances\Litasco supply"
FILENAME_PREFIX = "Europe Runs Recap"
# On accepte .xlsx et .xls (si .xls, xlrd doit être installé)
FILENAME_GLOB = f"{FILENAME_PREFIX} *.xls*"

# Fallback local (pièce jointe) si le partage n'est pas accessible
FALLBACK_FILE = Path("/mnt/data/Europe Runs Recap 09.16.2025.xlsx")


# =========================
# Utilitaires fichiers
# =========================
def _is_windows() -> bool:
    return platform.system() == "Windows"


def pick_latest_runs_file(base_dir: str | Path = RUNS_DIR) -> Path:
    """
    Retourne le fichier le plus récent correspondant au motif 'Europe Runs Recap *.xls*'.
    Si rien n'est trouvé (ou partage inaccessible), tente FALLBACK_FILE.
    """
    base = Path(base_dir)
    candidates: List[Path] = []

    try:
        if base.exists():
            # trie par date de modif décroissante
            candidates = sorted(
                base.glob(FILENAME_GLOB),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
    except Exception:
        # ex: UNC non monté sur Linux
        candidates = []

    if candidates:
        return candidates[0]

    if FALLBACK_FILE.exists():
        return FALLBACK_FILE

    raise FileNotFoundError(
        f"Aucun fichier trouvé dans {base_dir} avec le motif '{FILENAME_GLOB}' "
        f"et fallback absent : {FALLBACK_FILE}"
    )


# =========================
# Chargement & préparation
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def load_runs_workbook(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Lit le classeur et renvoie {sheet_name: DataFrame}, en nettoyant les lignes/colonnes vides.
    Supporte .xlsx (openpyxl) et .xls (xlrd si installé).
    """
    suffix = xlsx_path.suffix.lower()
    engine = None
    if suffix == ".xlsx":
        engine = "openpyxl"
    elif suffix == ".xls":
        # xlrd n'ouvre plus les .xlsx ; pour .xls il faut xlrd<=2.0
        engine = "xlrd"

    try:
        wb = pd.read_excel(xlsx_path, sheet_name=None, engine=engine)
    except Exception as e:
        if suffix == ".xls":
            raise RuntimeError(
                "Échec de lecture .xls. Assure-toi que 'xlrd' est installé et compatible pour les fichiers .xls."
            ) from e
        raise

    # Nettoyage léger
    cleaned: Dict[str, pd.DataFrame] = {}
    for sh, df in wb.items():
        df2 = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        cleaned[sh] = df2
    return cleaned


def _pick_region_sheet(wb: Dict[str, pd.DataFrame], region: str) -> Tuple[str, pd.DataFrame]:
    """
    region ∈ {'NWE','MED'}.
    Si des feuilles 'NWE'/'MED' existent -> on les prend.
    Sinon: 1ère feuille = NWE, 2ème = MED.
    """
    keys = list(wb.keys())
    if not keys:
        raise ValueError("Classeur vide : aucune feuille détectée.")

    lower_map = {k.lower(): k for k in keys}
    if region.upper() == "NWE":
        if "nwe" in lower_map:
            sh = lower_map["nwe"]
        else:
            sh = keys[0]
    else:
        if "med" in lower_map:
            sh = lower_map["med"]
        else:
            sh = keys[1] if len(keys) > 1 else keys[0]
    return sh, wb[sh]


def tidy_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format attendu :
      - 1ère colonne = dates mensuelles -> 'Date'
      - 1ère ligne = noms de pays pour les autres colonnes
    Sortie long format : [Date, Year, Month, MonthLabel, Country, Value]
    """
    if df.empty:
        return pd.DataFrame(columns=["Date", "Year", "Month", "MonthLabel", "Country", "Value"])

    df = df.copy()

    # Renommer la 1ère colonne en 'Date'
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Date"})

    # Parsing dates tolérant (Jan-2023, 2023-01, Jan23, etc.)
    def parse_date(x):
        if pd.isna(x):
            return np.nan
        try:
            return pd.to_datetime(x, errors="coerce", infer_datetime_format=True)
        except Exception:
            return pd.NaT

    df["Date"] = df["Date"].map(parse_date)
    df = df[df["Date"].notna()].copy()

    # Wide -> long
    value_cols = [c for c in df.columns if c != "Date"]
    long_df = df.melt(id_vars="Date", value_vars=value_cols, var_name="Country", value_name="Value")

    # Nettoyage / dérivées
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

    # Pivot index=Month, columns=Year, values=Value
    pivot = d.pivot_table(index="Month", columns="Year", values="Value", aggfunc="mean").sort_index()

    # Courbes par année
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

    # Choix automatique du fichier + override manuel
    with st.expander("📁 Fichier utilisé (auto: plus récent)"):
        st.caption(f"Dossier: {RUNS_DIR}")
        if not _is_windows() and (RUNS_DIR.startswith("\\") or RUNS_DIR.startswith("//")):
            st.warning(
                "Chemin UNC détecté sur un runtime non-Windows : peut ne pas être accessible.\n"
                "➡️ Monte le partage réseau ou utilise l’override ci-dessous."
            )
        override = st.text_input(
            "Override (chemin/nom de fichier .xls/.xlsx)",
            value="",
            help="Laisse vide pour auto-pick"
        )

        try:
            xlsx_path = Path(override) if override.strip() else pick_latest_runs_file(RUNS_DIR)
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

    # (Optionnel) filtrage par années — décommente au besoin
    # years = sorted(long_df["Year"].unique().tolist())
    # sel_years = st.multiselect("Années", years, default=years)
    # if sel_years:
    #     long_df = long_df[long_df["Year"].isin(sel_years)]

    # Affichage en grille 3 colonnes
    cols = st.columns(3)
    for i, country in enumerate(selected):
        fig = plot_country_seasonal(long_df, country, unit=unit)
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True, key=f"runs_{region}_{country}_{i}")
