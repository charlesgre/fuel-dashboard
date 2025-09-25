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
RUNS_DIR = r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\Litasco balances\Litasco supply"
REPO_RUNS_DIR = Path(__file__).resolve().parent / "Litasco balances" / "Litasco supply"
FALLBACK_FILE = Path("/mnt/data/Europe Runs Recap 09.16.2025.xlsx")
FILENAME_PREFIX = "Europe Runs Recap"
FILENAME_GLOB = f"{FILENAME_PREFIX} *.xls*"

# Années à afficher
YEAR_MIN, YEAR_MAX = 2020, 2026
# Années utilisées pour le "range" (ruban min–max)
BASE_RANGE_YEARS = [2020, 2021, 2022, 2023, 2024]

# Couleurs mises en avant
HIGHLIGHT_COLORS = {
    2024: "#2ca02c",  # vert
    2025: "#000000",  # noir
    2026: "#d62728",  # rouge
}
# Palette douce pour les années ≤ 2023
MUTED_PALETTE = ["#9ecae1", "#c7e9c0", "#fdd0a2", "#bcbddc", "#fdae6b", "#bdbdbd", "#c7c7c7"]

# =========================
# Utilitaires fichiers
# =========================
def _is_windows() -> bool:
    return platform.system() == "Windows"

def _latest_in(folder: Path) -> Path | None:
    try:
        if folder.exists():
            files = sorted(folder.glob(FILENAME_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return files[0]
    except Exception:
        return None
    return None

def pick_latest_runs_file() -> Path:
    """Ordre de recherche: UNC -> repo local -> fallback /mnt/data."""
    p = _latest_in(Path(RUNS_DIR))
    if p:
        return p
    p = _latest_in(REPO_RUNS_DIR)
    if p:
        return p
    if FALLBACK_FILE.exists():
        return FALLBACK_FILE
    raise FileNotFoundError(
        f"Aucun fichier {FILENAME_GLOB} trouvé dans:\n- UNC: {RUNS_DIR}\n- Repo: {REPO_RUNS_DIR}\n"
        f"et fallback absent: {FALLBACK_FILE}"
    )

# =========================
# Chargement & préparation
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def load_runs_workbook(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    """Lit le classeur et renvoie {sheet_name: DataFrame} nettoyés."""
    suffix = xlsx_path.suffix.lower()
    engine = "openpyxl" if suffix == ".xlsx" else ("xlrd" if suffix == ".xls" else None)
    try:
        wb = pd.read_excel(xlsx_path, sheet_name=None, engine=engine)
    except Exception as e:
        if suffix == ".xls":
            raise RuntimeError("Lecture .xls impossible (installe xlrd compatible .xls ou convertis en .xlsx).") from e
        raise
    return {sh: df.dropna(axis=0, how="all").dropna(axis=1, how="all") for sh, df in wb.items()}

def _pick_region_sheet(wb: Dict[str, pd.DataFrame], region: str) -> Tuple[str, pd.DataFrame]:
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
    """Wide -> long + colonnes Year/Month/MonthLabel; filtre 2020→2026."""
    if df.empty:
        return pd.DataFrame(columns=["Date", "Year", "Month", "MonthLabel", "Country", "Value"])

    df = df.copy()
    df = df.rename(columns={df.columns[0]: "Date"})

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
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ordered=True,
    )

    long_df = long_df[(long_df["Year"] >= YEAR_MIN) & (long_df["Year"] <= YEAR_MAX)]
    return long_df.sort_values(["Country", "Year", "Month"]).reset_index(drop=True)

# =========================
# Tracés
# =========================
def _style_for_year(year: int, idx_other: int) -> dict:
    """Couleur/épaisseur/opacité selon l'année."""
    if year in HIGHLIGHT_COLORS:
        return {"color": HIGHLIGHT_COLORS[year], "width": 3.0, "opacity": 1.0, "marker": 6}
    # ≤ 2023 : un peu plus épais que les autres muted, mais estompés
    color = MUTED_PALETTE[idx_other % len(MUTED_PALETTE)]
    width = 1.8 if year <= 2023 else 1.5
    return {"color": color, "width": width, "opacity": 0.55, "marker": 4}

def _add_range_band(fig: go.Figure, d: pd.DataFrame, base_years: List[int]) -> None:
    """
    Ajoute un ruban min–max (range) calculé sur base_years, par mois.
    Tracé: une courbe 'min' (invisible) + une courbe 'max' avec fill='tonexty'.
    """
    d_base = d[d["Year"].isin(base_years)]
    if d_base.empty:
        return

    # Tableau 12 mois pour min/max (avec reindex 1..12)
    pivot = d_base.pivot_table(index="Month", columns="Year", values="Value", aggfunc="mean")
    months = list(range(1, 13))
    min_vals = pivot.reindex(months).min(axis=1).values
    max_vals = pivot.reindex(months).max(axis=1).values

    band_color = "rgba(31,119,180,0.14)"   # bleu doux transparent
    edge_color = "rgba(31,119,180,0.40)"

    # bas (pas dans la légende)
    fig.add_trace(go.Scatter(
        x=months, y=min_vals, mode="lines", line=dict(color=edge_color, width=0.5),
        hoverinfo="skip", showlegend=False, name="min 2020–2024"
    ))
    # haut (affiche la légende + remplissage)
    fig.add_trace(go.Scatter(
        x=months, y=max_vals, mode="lines", line=dict(color=edge_color, width=0.5),
        fill="tonexty", fillcolor=band_color,
        name="Range 2020–2024", hoverinfo="skip"
    ))

def plot_country_seasonal(long_df: pd.DataFrame, country: str, unit: str | None = None) -> go.Figure:
    d = long_df[long_df["Country"] == country]
    fig = go.Figure()

    if d.empty:
        fig.update_layout(title=f"{country} — (aucune donnée)", template="plotly_white", height=300)
        return fig

    # Ruban de range 2020–2024 (dessiné d'abord pour rester en arrière-plan)
    _add_range_band(fig, d, BASE_RANGE_YEARS)

    # Lignes par année (dans l'ordre croissant)
    years = sorted(d["Year"].unique().tolist())
    other_idx = 0
    for y in years:
        sty = _style_for_year(y, other_idx)
        if y not in HIGHLIGHT_COLORS:
            other_idx += 1
        y_data = d[d["Year"] == y].set_index("Month").reindex(range(1, 13))["Value"].values
        fig.add_trace(go.Scatter(
            x=list(range(1, 13)),
            y=y_data,
            mode="lines+markers",
            name=str(y),
            connectgaps=True,
            line=dict(color=sty["color"], width=sty["width"]),
            marker=dict(size=sty["marker"], color=sty["color"]),
            opacity=sty["opacity"],
        ))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        showgrid=False, zeroline=False,
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

    # Région
    region = st.radio("Région", ["NWE", "MED"], horizontal=True)

    # Fichier (auto)
    try:
        xlsx_path = pick_latest_runs_file()
        st.caption(f"Fichier utilisé : **{xlsx_path.name}** — Zone ombrée = range 2020–2024 ; années affichées 2020–2026.")
    except Exception as e:
        st.error(str(e)); st.stop()

    # Chargement & sélection de feuille
    try:
        wb = load_runs_workbook(xlsx_path)
    except Exception as e:
        st.exception(e); st.stop()

    sheet_name, df_region = _pick_region_sheet(wb, region)
    st.caption(f"Feuille utilisée pour {region} : **{sheet_name}**")

    # Mise en forme
    long_df = tidy_runs_df(df_region)

    # Unité + pays
    unit = st.text_input("Unité (axe Y)", value="", help="ex: kb/d (optionnel)")
    all_countries = sorted(long_df["Country"].unique().tolist())
    selected = st.multiselect("Pays à afficher", all_countries, default=all_countries)

    # Affichage en grille
    cols = st.columns(3)
    for i, country in enumerate(selected):
        fig = plot_country_seasonal(long_df, country, unit=unit)
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True, key=f"runs_{region}_{country}_{i}")
