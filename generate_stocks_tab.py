# -*- coding: utf-8 -*-

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

SHEET_CANDIDATES = ["Query1", "Sheet1"]  # ordre d’essai

TITLE_MAP = {
    "Insights Global Residuals ARA Stocks Weekly": "ARA stocks",
    "EIA 9 Resid FO PADD3 Stks": "PADD 3 stocks",
    "FEDCom Platts Fujairah Heavy Distillates and Residues Stocks Volume": "Fujairah stocks",
    "Enterprise Singapore Residues Singapore Stocks": "Singapore stocks",
}

COLOR_2025 = "#d62728"  # rouge
COLOR_2024 = "#2ca02c"  # vert
COLOR_2023 = "#bcbd22"  # jaune
OTHER_ALPHA = 0.35

HIST_START = 2015
HIST_END   = 2024   # inclus


# ----------------- Data -----------------

@st.cache_data(show_spinner=False)
def _read_excel(excel_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)

    sheet_name = next((s for s in SHEET_CANDIDATES if s in xls.sheet_names), None)
    if sheet_name is None:
        for s in xls.sheet_names:
            tmp = pd.read_excel(excel_path, sheet_name=s)
            if not tmp.empty:
                sheet_name = s
                break
    if sheet_name is None:
        raise ValueError("Aucune feuille non vide trouvée dans l'Excel.")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    needed = {"DESCRIPTION", "ASSESSDATE", "VALUE"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Colonnes manquantes dans '{sheet_name}': {miss}")

    keep_cols = ["DESCRIPTION", "ASSESSDATE", "VALUE"]
    if "UOM" in df.columns:
        keep_cols.append("UOM")

    df = df[keep_cols].copy()
    df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"], errors="coerce")
    df = df.dropna(subset=["ASSESSDATE"])
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["VALUE"])

    df = df[df["DESCRIPTION"].isin(TITLE_MAP.keys())].copy()
    df["TITLE"] = df["DESCRIPTION"].map(TITLE_MAP)
    if "UOM" not in df.columns:
        df["UOM"] = ""

    df["Year"] = df["ASSESSDATE"].dt.year
    df["Week"] = df["ASSESSDATE"].dt.isocalendar().week.astype(int)
    return df


def load_data() -> pd.DataFrame:
    # === Chargement et prépa des données ===
    excel_path = os.path.join("Stocks", "Data global stocks.xlsx")
    if not os.path.exists(excel_path):
        st.error(f"Fichier introuvable : {excel_path}\n"
                 "Vérifie que le fichier est bien dans le dossier du dashboard.")
        st.stop()
    return _read_excel(excel_path)


# ----------------- Stats hebdo -----------------

def weekly_stats(hist_df: pd.DataFrame):
    pivot = hist_df.groupby(["Week", "Year"])["VALUE"].mean().unstack()
    pivot = pivot.reindex(pd.Index(range(1, 54), name="Week"))
    mean_vals = pivot.mean(axis=1, skipna=True)
    min_vals  = pivot.min(axis=1, skipna=True)
    max_vals  = pivot.max(axis=1, skipna=True)
    return mean_vals, min_vals, max_vals


# ----------------- Table variations -----------------

def _nearest_on_or_before(series: pd.Series, target_date: pd.Timestamp):
    s = series[series.index <= target_date]
    if s.empty:
        return None, None
    idx = s.index.max()
    return idx, s.loc[idx]


def compute_change_table(df_region: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ts = (
        df_region.sort_values("ASSESSDATE")
        .set_index("ASSESSDATE")["VALUE"]
        .asfreq("D", method="pad")
    )
    latest_date = ts.dropna().index.max()
    latest_val  = float(ts.loc[latest_date])

    week_ago  = latest_date - pd.DateOffset(weeks=1)
    month_ago = latest_date - pd.DateOffset(months=1)
    year_ago  = latest_date - pd.DateOffset(years=1)

    d_w, v_w = _nearest_on_or_before(ts, week_ago)
    d_m, v_m = _nearest_on_or_before(ts, month_ago)
    d_y, v_y = _nearest_on_or_before(ts, year_ago)

    rows = []
    def add_row(label, d, v, base):
        if d is None or v is None:
            rows.append([label, "-", "-", "-", "-"])
        else:
            chg = base - v
            pct = (chg / v) * 100 if v != 0 else np.nan
            rows.append([
                label,
                d.strftime("%d-%m-%Y"),
                f"{v:,.2f}",
                f"{chg:,.2f}",
                f"{pct:,.2f}%" if np.isfinite(pct) else "-"
            ])

    uom = df_region["UOM"].mode().iloc[0] if not df_region["UOM"].empty else ""
    rows.insert(0, ["Latest", latest_date.strftime("%d-%m-%Y"), f"{latest_val:,.2f}", "-", "-"])
    add_row("Previous week", d_w, v_w, latest_val)
    add_row("Previous month", d_m, v_m, latest_val)
    add_row("Previous year", d_y, v_y, latest_val)

    table_df = pd.DataFrame(
        rows,
        columns=["Label", "Date",
                 f"Value ({uom})" if uom else "Value",
                 f"Change ({uom})" if uom else "Change",
                 "Change %"]
    )
    return table_df, uom


# ----------------- Plotly chart -----------------

def build_plotly_chart(data: pd.DataFrame, title: str) -> go.Figure:
    hist = data[(data["Year"] >= HIST_START) & (data["Year"] <= HIST_END)].copy()
    mean_vals, min_vals, max_vals = weekly_stats(hist)
    x = mean_vals.index.values

    fig = go.Figure()

    # Bande min-max
    fig.add_trace(go.Scatter(
        x=x, y=min_vals.values, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=max_vals.values, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(128,128,128,0.30)",
        name=f"Range {HIST_START}–{HIST_END}"
    ))

    # Moyenne
    fig.add_trace(go.Scatter(
        x=x, y=mean_vals.values, mode="lines",
        line=dict(color="black", dash="dash", width=2),
        name=f"Moyenne {HIST_START}–{HIST_END}"
    ))

    # Séries par année
    for year, grp in data.groupby("Year"):
        weekly_vals = grp.groupby("Week")["VALUE"].mean().reindex(range(1, 54))
        line_color = None
        opacity = 1.0
        if year == 2025:
            line_color = COLOR_2025
        elif year == 2024:
            line_color = COLOR_2024
        elif year == 2023:
            line_color = COLOR_2023
        else:
            opacity = OTHER_ALPHA

        fig.add_trace(go.Scatter(
            x=weekly_vals.index, y=weekly_vals.values,
            mode="lines", name=str(year),
            line=dict(color=line_color, width=2) if line_color else dict(width=2),
            opacity=opacity, connectgaps=True
        ))

    fig.update_layout(
        title=f"Stock saisonnier hebdomadaire ({HIST_START}–{dt.date.today().year}) – {title}",
        xaxis_title="Semaine",
        yaxis_title="Stocks (KT)",
        hovermode="x unified",
        legend_title="Année",
        margin=dict(l=30, r=10, t=60, b=30),
        height=520,
    )
    fig.update_xaxes(range=[1, 53], dtick=4, showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# ----------------- Rendu onglet -----------------

def render_tab():
    st.subheader("Global stocks (interactif)")

    df = load_data()
    df = df[df["Year"] >= 2015].copy()

    regions = sorted(df["TITLE"].unique())
    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.multiselect("Régions", regions, default=regions)
    with col2:
        min_year = int(df["Year"].min())
        max_year = int(df["Year"].max())
        years = st.slider("Années à afficher", min_year, max_year, (max_year-5 if max_year-5>=min_year else min_year, max_year))

    df = df[df["TITLE"].isin(selected)]
    df = df[(df["Year"] >= years[0]) & (df["Year"] <= years[1])]

    for title in selected:
        temp = df[df["TITLE"] == title].copy()
        st.plotly_chart(build_plotly_chart(temp, title), use_container_width=True)

        table_df, uom = compute_change_table(temp[["ASSESSDATE", "VALUE", "UOM"]].copy())
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        csv = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Télécharger la table – {title} (CSV)",
            data=csv, file_name=f"change_table_{title.replace(' ','_')}.csv",
            mime="text/csv"
        )
        st.markdown("---")
