# litasco_runs.py
from __future__ import annotations

import platform
import re
from datetime import datetime
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
FILENAME_PREFIX = "Europe Runs Recap"
FILENAME_GLOB = f"{FILENAME_PREFIX} *.xls*"

# Années à afficher
YEAR_MIN, YEAR_MAX = 2020, 2026
BASE_RANGE_YEARS = [2020, 2021, 2022, 2023, 2024]

# Couleurs mises en avant
HIGHLIGHT_COLORS = {2024: "#2ca02c", 2025: "#000000", 2026: "#d62728"}
MUTED_PALETTE = ["#9ecae1", "#c7e9c0", "#fdd0a2", "#bcbddc", "#fdae6b", "#bdbdbd", "#c7c7c7"]

# =========================
# Utilitaires fichiers
# =========================
def _is_windows() -> bool:
    return platform.system() == "Windows"

# Nouvelles regex : on supporte YYYY.MM.DD / YYYY-MM-DD ET MM.DD.YYYY / MM-DD-YYYY
_DATE_RE_YMD = re.compile(r"(\d{4})[.\-](\d{2})[.\-](\d{2})")  # 2025.10.06 ou 2025-10-06
_DATE_RE_MDY = re.compile(r"(\d{2})[.\-](\d{2})[.\-](\d{4})")  # 10.06.2025 ou 10-06-2025

def _date_from_filename(p: Path) -> datetime | None:
    """
    Extrait une date depuis le nom de fichier.
    Priorité au format YYYY.MM.DD / YYYY-MM-DD, puis fallback MM.DD.YYYY / MM-DD-YYYY.
    """
    s = p.name
    m = _DATE_RE_YMD.search(s)
    if m:
        yyyy, mm, dd = map(int, m.groups())
        try:
            return datetime(yyyy, mm, dd)
        except ValueError:
            return None
    m = _DATE_RE_MDY.search(s)
    if m:
        mm, dd, yyyy = map(int, m.groups())
        try:
            return datetime(yyyy, mm, dd)
        except ValueError:
            return None
    return None

def _sort_key(p: Path):
    """Tri prioritaire par date du nom si trouvée, sinon très ancien; puis par mtime."""
    dt_name = _date_from_filename(p) or datetime(1900, 1, 1)
    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0
    return (dt_name, mtime)

def _candidates_in(folder: Path) -> List[Path]:
    """Liste des fichiers candidats d’un dossier (sans ~$.)."""
    if not folder.exists():
        return []
    return [p for p in Path(folder).glob(FILENAME_GLOB) if not p.name.startswith("~$")]

def pick_latest_runs_file(debug: bool = False) -> Path:
    """
    Agrège UNC + repo local et retourne le fichier le plus récent selon :
    1) date dans le nom (YYYY.MM.DD / YYYY-MM-DD prioritaire; compat MM.DD.YYYY / MM-DD-YYYY)
    2) mtime
    """
    folders = [Path(RUNS_DIR), REPO_RUNS_DIR]
    candidates: List[Path] = []
    for f in folders:
        candidates.extend(_candidates_in(f))

    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier {FILENAME_GLOB} trouvé dans:\n- UNC: {RUNS_DIR}\n- Repo: {REPO_RUNS_DIR}"
        )

    # Tri par (date extraite, mtime)
    candidates.sort(key=_sort_key, reverse=True)
    best = candidates[0]

    if debug:
        rows = []
        for p in candidates:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except Exception:
                mtime = None
            dt = _date_from_filename(p)
            rows.append(
                {
                    "name": p.name,
                    "folder": str(p.parent),
                    "date_in_name": dt.strftime("%Y-%m-%d") if dt else "",
                    "mtime": mtime.strftime("%Y-%m-%d %H:%M:%S") if mtime else "",
                }
            )
        st.write("### Fichiers candidats (triés par date nom puis mtime)")
        st.dataframe(pd.DataFrame(rows))

    return best

# =========================
# Chargement & préparation
# =========================
@st.cache_data(show_spinner=False, ttl=3600, hash_funcs={Path: lambda p: (p.as_posix(), p.stat().st_mtime)})
def load_runs_workbook(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    """Lit le classeur et renvoie {sheet_name: DataFrame} nettoyés."""
    suffix = xlsx_path.suffix.lower()
    if suffix == ".xlsx":
        engine = "openpyxl"
    elif suffix == ".xls":
        engine = "xlrd"
    elif suffix == ".xlsb":
        engine = "pyxlsb"
    else:
        engine = None

    try:
        wb = pd.read_excel(xlsx_path, sheet_name=None, engine=engine)
    except Exception as e:
        if suffix == ".xls":
            raise RuntimeError("Lecture .xls impossible (installe xlrd compatible .xls ou convertis en .xlsx).") from e
        if suffix == ".xlsb":
            raise RuntimeError("Lecture .xlsb impossible (installe pyxlsb: pip install pyxlsb).") from e
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
    if year in HIGHLIGHT_COLORS:
        return {"color": HIGHLIGHT_COLORS[year], "width": 3.0, "opacity": 1.0, "marker": 6}
    color = MUTED_PALETTE[idx_other % len(MUTED_PALETTE)]
    width = 1.8 if year <= 2023 else 1.5
    return {"color": color, "width": width, "opacity": 0.55, "marker": 4}

def _add_range_band(fig: go.Figure, d: pd.DataFrame, base_years: List[int]) -> None:
    d_base = d[d["Year"].isin(base_years)]
    if d_base.empty:
        return
    pivot = d_base.pivot_table(index="Month", columns="Year", values="Value", aggfunc="mean")
    months = list(range(1, 13))
    min_vals = pivot.reindex(months).min(axis=1).values
    max_vals = pivot.reindex(months).max(axis=1).values
    band_color = "rgba(31,119,180,0.14)"
    edge_color = "rgba(31,119,180,0.40)"
    fig.add_trace(go.Scatter(x=months, y=min_vals, mode="lines", line=dict(color=edge_color, width=0.5),
                             hoverinfo="skip", showlegend=False, name="min 2020–2024"))
    fig.add_trace(go.Scatter(x=months, y=max_vals, mode="lines", line=dict(color=edge_color, width=0.5),
                             fill="tonexty", fillcolor=band_color, name="Range 2020–2024", hoverinfo="skip"))

def plot_country_seasonal(long_df: pd.DataFrame, country: str, unit: str | None = None) -> go.Figure:
    d = long_df[long_df["Country"] == country]
    fig = go.Figure()
    if d.empty:
        fig.update_layout(title=f"{country} — (aucune donnée)", template="plotly_white", height=300)
        return fig
    _add_range_band(fig, d, BASE_RANGE_YEARS)
    years = sorted(d["Year"].unique().tolist())
    other_idx = 0
    for y in years:
        sty = _style_for_year(y, other_idx)
        if y not in HIGHLIGHT_COLORS:
            other_idx += 1
        y_data = d[d["Year"] == y].set_index("Month").reindex(range(1, 13))["Value"].values
        fig.add_trace(go.Scatter(x=list(range(1, 13)), y=y_data, mode="lines+markers", name=str(y),
                                 connectgaps=True, line=dict(color=sty["color"], width=sty["width"]),
                                 marker=dict(size=sty["marker"], color=sty["color"]), opacity=sty["opacity"]))
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                     showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=(unit or ""), rangemode="tozero")
    fig.update_layout(title=country, legend_title_text="Année", template="plotly_white",
                      height=320, margin=dict(l=10, r=10, t=40, b=10))
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

    # Debug (optionnel) pour voir la liste détectée
    debug_files = st.toggle("Afficher la liste des fichiers détectés (debug)")

    # Fichier (auto)
    try:
        xlsx_path = pick_latest_runs_file(debug=debug_files)
        st.caption(
            f"Fichier utilisé : **{xlsx_path.name}** (dossier : {xlsx_path.parent}) — "
            "Zone ombrée = range 2020–2024 ; années affichées 2020–2026."
        )
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
