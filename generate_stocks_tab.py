# -*- coding: utf-8 -*-

import os, re, datetime as dt
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Paramètres ---
SHEET_CANDIDATES = ["Query1", "Sheet1", "Data", "Sheet", "Feuil1", "Platts window"]
TITLE_MAP = {
    "Insights Global Residuals ARA Stocks Weekly": "ARA stocks",
    "EIA 9 Resid FO PADD3 Stks": "PADD 3 stocks",
    "FEDCom Platts Fujairah Heavy Distillates and Residues Stocks Volume": "Fujairah stocks",
    "Enterprise Singapore Residues Singapore Stocks": "Singapore stocks",
}
COLOR_2025, COLOR_2024, COLOR_2023, OTHER_ALPHA = "#d62728", "#2ca02c", "#bcbd22", 0.35
HIST_START, HIST_END = 2015, 2024

# --- Helpers de normalisation / alias ---
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

DESC_ALIASES  = {"description","desc","series","name","instrument","instrumentname","title","libelle"}
DATE_ALIASES  = {"assessdate","date","assessmentdate","tradedate","pricedate","asofdate","datetime"}
VALUE_ALIASES = {"value","val","close","price","obs","volume","stock","stocks","kt","valeur"}
UOM_ALIASES   = {"uom","unit","units","measure","unite"}

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm_map = {_norm(c): c for c in df.columns}

    def pick(aliases):
        for a in aliases:
            if a in norm_map: return norm_map[a]
        return None

    c_desc  = pick(DESC_ALIASES)
    c_date  = pick(DATE_ALIASES)
    c_value = pick(VALUE_ALIASES)
    c_uom   = pick(UOM_ALIASES)

    missing = [k for k,v in {"DESCRIPTION":c_desc, "ASSESSDATE":c_date, "VALUE":c_value}.items() if v is None]
    if missing:
        raise ValueError(
            f"Colonnes non reconnues: {missing}\n"
            f"Colonnes vues: {list(df.columns)}"
        )

    out = df[[c_desc, c_date, c_value] + ([c_uom] if c_uom else [])].rename(columns={
        c_desc:"DESCRIPTION", c_date:"ASSESSDATE", c_value:"VALUE", **({c_uom:"UOM"} if c_uom else {})
    })
    if "UOM" not in out.columns: out["UOM"] = ""
    return out

def _try_read_with_header_row(path: str, sheet: str, header_row: int) -> pd.DataFrame:
    tmp = pd.read_excel(path, sheet_name=sheet, header=header_row)
    tmp.columns = pd.io.parsers.ParserBase({'names':tmp.columns})._maybe_dedup_names(tmp.columns)
    return _map_columns(tmp)

@st.cache_data(show_spinner=False)
def _read_excel(path: str) -> tuple[pd.DataFrame, str, int]:
    xls = pd.ExcelFile(path)
    sheets = [s for s in SHEET_CANDIDATES if s in xls.sheet_names] + [s for s in xls.sheet_names if s not in SHEET_CANDIDATES]

    for sheet in sheets:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=50)

        for hdr in range(min(50, len(raw))):
            candidate = raw.iloc[hdr].astype(str).tolist()
            normed = {_norm(c) for c in candidate}
            hit_count = (
                (len(normed & DESC_ALIASES) > 0) +
                (len(normed & DATE_ALIASES) > 0) +
                (len(normed & VALUE_ALIASES) > 0)
            )
            if hit_count >= 2:  # plus tolérant (2 suffisent)
                try:
                    df = _try_read_with_header_row(path, sheet, hdr)
                    return df, sheet, hdr
                except Exception:
                    continue

        # fallback header=0
        try:
            df = _try_read_with_header_row(path, sheet, 0)
            return df, sheet, 0
        except Exception:
            pass

    raise ValueError(
        "Impossible de trouver une feuille avec les colonnes Description / Date / Value.\n"
        f"Feuilles disponibles: {xls.sheet_names}"
    )

def load_data() -> pd.DataFrame:
    excel_path = os.path.join("Platts window", "Window platts global data.xlsx")
    if not os.path.exists(excel_path):
        st.error(f"Fichier introuvable : {excel_path}")
        st.stop()
    df, sheet, hdr = _read_excel(excel_path)

    df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"], errors="coerce")
    df = df.dropna(subset=["ASSESSDATE"])
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["VALUE"])

    def map_title(desc: str) -> str | None:
        if desc in TITLE_MAP:
            return TITLE_MAP[desc]
        d = str(desc).lower()
        if ("resid" in d or "residual" in d) and "ara" in d: return "ARA stocks"
        if "padd" in d and "3" in d: return "PADD 3 stocks"
        if "fujairah" in d or "fedcom" in d: return "Fujairah stocks"
        if "singapore" in d: return "Singapore stocks"
        return None

    df["TITLE"] = df["DESCRIPTION"].map(map_title)
    df = df[df["TITLE"].notna()].copy()

    df["Year"] = df["ASSESSDATE"].dt.year
    df["Week"] = df["ASSESSDATE"].dt.isocalendar().week.astype(int)

    st.caption(f"Feuille détectée: **{sheet}** — Ligne d’entête: **{hdr+1}**")
    return df

# --- Stats hebdo ---
def weekly_stats(hist_df: pd.DataFrame):
    pivot = hist_df.groupby(["Week","Year"])["VALUE"].mean().unstack()
    pivot = pivot.reindex(pd.Index(range(1,54), name="Week"))
    return pivot.mean(axis=1, skipna=True), pivot.min(axis=1, skipna=True), pivot.max(axis=1, skipna=True)

# --- Table variations ---
def _nearest_on_or_before(series: pd.Series, target_date: pd.Timestamp):
    s = series[series.index <= target_date]
    if s.empty: return None, None
    idx = s.index.max(); return idx, s.loc[idx]

def compute_change_table(df_region: pd.DataFrame):
    ts = df_region.sort_values("ASSESSDATE").set_index("ASSESSDATE")["VALUE"].asfreq("D", method="pad")
    latest_date = ts.dropna().index.max(); latest_val = float(ts.loc[latest_date])
    d_w, v_w = _nearest_on_or_before(ts, latest_date - pd.DateOffset(weeks=1))
    d_m, v_m = _nearest_on_or_before(ts, latest_date - pd.DateOffset(months=1))
    d_y, v_y = _nearest_on_or_before(ts, latest_date - pd.DateOffset(years=1))
    rows = [["Latest", latest_date.strftime("%d-%m-%Y"), f"{latest_val:,.2f}", "-", "-"]]
    def add(label,d,v):
        if d is None or v is None: rows.append([label,"-","-","-","-"]); return
        chg = latest_val - v; pct = (chg / v) * 100 if v else np.nan
        rows.append([label, d.strftime("%d-%m-%Y"), f"{v:,.2f}", f"{chg:,.2f}", f"{pct:,.2f}%" if np.isfinite(pct) else "-"])
    add("Previous week", d_w, v_w); add("Previous month", d_m, v_m); add("Previous year", d_y, v_y)
    uom = df_region["UOM"].mode().iloc[0] if not df_region["UOM"].empty else ""
    table = pd.DataFrame(rows, columns=["Label","Date", f"Value ({uom})" if uom else "Value", f"Change ({uom})" if uom else "Change","Change %"])
    return table, uom

# --- Plotly chart ---
def build_plotly_chart(data: pd.DataFrame, title: str) -> go.Figure:
    hist = data[(data["Year"] >= HIST_START) & (data["Year"] <= HIST_END)]
    mean_vals, min_vals, max_vals = weekly_stats(hist); x = mean_vals.index.values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=min_vals.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=max_vals.values, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(128,128,128,0.30)", name=f"Range {HIST_START}–{HIST_END}"))
    fig.add_trace(go.Scatter(x=x, y=mean_vals.values, mode="lines",
                             line=dict(color="black", dash="dash", width=2), name=f"Moyenne {HIST_START}–{HIST_END}"))
    for year, grp in data.groupby("Year"):
        weekly_vals = grp.groupby("Week")["VALUE"].mean().reindex(range(1,54))
        color, opacity = (COLOR_2025,1.0) if year==2025 else (COLOR_2024,1.0) if year==2024 else (COLOR_2023,1.0) if year==2023 else (None, OTHER_ALPHA)
        fig.add_trace(go.Scatter(x=weekly_vals.index, y=weekly_vals.values, mode="lines", name=str(year),
                                 line=dict(color=color, width=2) if color else dict(width=2),
                                 opacity=opacity, connectgaps=True))
    fig.update_layout(title=f"Stock saisonnier hebdomadaire ({HIST_START}–{dt.date.today().year}) – {title}",
                      xaxis_title="Semaine", yaxis_title="Stocks (KT)", hovermode="x unified",
                      legend_title="Année", margin=dict(l=30,r=10,t=60,b=30), height=520)
    fig.update_xaxes(range=[1,53], dtick=4, showgrid=True); fig.update_yaxes(showgrid=True)
    return fig

# --- Onglet ---
def generate_stocks_tab():
    st.subheader("Global stocks (interactif)")
    df = load_data()
    df = df[df["Year"] >= 2015].copy()

    regions = sorted(df["TITLE"].unique())
    c1, c2 = st.columns([2,1])
    with c1:
        selected = st.multiselect("Régions", regions, default=regions)
    with c2:
        miny, maxy = int(df["Year"].min()), int(df["Year"].max())
        years = st.slider("Années à afficher", miny, maxy, (max(miny, maxy-5), maxy))

    df = df[df["TITLE"].isin(selected) & df["Year"].between(years[0], years[1])]

    for title in selected:
        temp = df[df["TITLE"] == title]
        st.plotly_chart(build_plotly_chart(temp, title), use_container_width=True)
        table_df, _ = compute_change_table(temp[["ASSESSDATE","VALUE","UOM"]])
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Télécharger la table – {title} (CSV)",
            data=table_df.to_csv(index=False).encode("utf-8"),
            file_name=f"change_table_{title.replace(' ','_')}.csv",
            mime="text/csv",
        )
        st.markdown("---")
