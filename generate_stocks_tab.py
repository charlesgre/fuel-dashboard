# -*- coding: utf-8 -*-

import os, re, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ========= CONFIG =========
EXCEL_REL_PATH = os.path.join("Stocks", "Data global stocks.xlsx")
SHEET_CANDIDATES = ["Query1", "Sheet1", "Data", "Sheet", "Feuil1"]

TITLE_MAP = {
    "Insights Global Residuals ARA Stocks Weekly": "ARA stocks",
    "EIA 9 Resid FO PADD3 Stks": "PADD 3 stocks",
    "FEDCom Platts Fujairah Heavy Distillates and Residues Stocks Volume": "Fujairah stocks",
    "Enterprise Singapore Residues Singapore Stocks": "Singapore stocks",
}

COLOR_2025, COLOR_2024, COLOR_2023 = "#d62728", "#2ca02c", "#bcbd22"
OTHER_ALPHA = 0.35
HIST_START, HIST_END = 2015, 2024


# ---------- helpers: tolerant column mapping ----------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

def _find_col(cols, keywords):
    cols = list(cols)
    normed = {_norm(c): c for c in cols}
    # 1) exact match on normalized
    for kw in keywords:
        if kw in normed:
            return normed[kw]
    # 2) substring match on normalized names
    for c in cols:
        nc = _norm(c)
        if any(kw in nc for kw in keywords):
            return c
    return None

DESC_KEYS  = ["description", "desc", "series", "name", "instrument", "title"]
DATE_KEYS  = ["assessdate", "date", "assessmentdate", "tradedate", "pricedate", "datetime", "asofdate"]
VALUE_KEYS = ["value", "val", "close", "price", "obs", "volume", "stock", "stocks", "qty", "quantity"]
UOM_KEYS   = ["uom", "unit", "units", "measure", "unite"]

def _map_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    c_desc  = _find_col(cols, DESC_KEYS)
    c_date  = _find_col(cols, DATE_KEYS)
    c_value = _find_col(cols, VALUE_KEYS)
    c_uom   = _find_col(cols, UOM_KEYS)

    missing = [name for name, c in [("DESCRIPTION", c_desc), ("ASSESSDATE", c_date), ("VALUE", c_value)] if c is None]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    out = df[[c_desc, c_date, c_value] + ([c_uom] if c_uom else [])].rename(
        columns={
            c_desc: "DESCRIPTION",
            c_date: "ASSESSDATE",
            c_value: "VALUE",
            **({c_uom: "UOM"} if c_uom else {}),
        }
    )
    if "UOM" not in out.columns:
        out["UOM"] = ""
    return out


# ---------- data loading (simple & robust) ----------
@st.cache_data(show_spinner=False)
def _read_excel_all(path: str) -> tuple[pd.DataFrame, str]:
    xls = pd.ExcelFile(path)
    sheet_order = [s for s in SHEET_CANDIDATES if s in xls.sheet_names] + \
                  [s for s in xls.sheet_names if s not in SHEET_CANDIDATES]

    last_error = None
    for sheet in sheet_order:
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=0)
            if raw.empty:
                continue
            # ensure string column names
            raw.columns = [str(c) for c in raw.columns]
            df = _map_required_columns(raw)

            # clean types
            df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"], errors="coerce")
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
            df = df.dropna(subset=["ASSESSDATE", "VALUE"])

            # map titles (tolerant)
            def map_title(d):
                if d in TITLE_MAP: return TITLE_MAP[d]
                s = str(d).lower()
                if ("resid" in s or "residual" in s) and "ara" in s: return "ARA stocks"
                if "padd" in s and "3" in s: return "PADD 3 stocks"
                if "fujairah" in s or "fedcom" in s: return "Fujairah stocks"
                if "singapore" in s: return "Singapore stocks"
                return None

            df["TITLE"] = df["DESCRIPTION"].map(map_title)
            df = df[df["TITLE"].notna()].copy()
            if df.empty:
                continue

            # add year/week
            df["Year"] = df["ASSESSDATE"].dt.year
            df["Week"] = df["ASSESSDATE"].dt.isocalendar().week.astype(int)

            return df, sheet
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise ValueError(f"Unable to parse any sheet in {os.path.basename(path)}. "
                         f"Sheets: {xls.sheet_names}. Last error: {last_error}")
    raise ValueError(f"No usable sheet found in {os.path.basename(path)}. Sheets: {xls.sheet_names}")

def load_data() -> pd.DataFrame:
    path = os.path.join("Stocks", "Data global stocks.xlsx")  # <- your path
    if not os.path.exists(path):
        st.error(f"Excel not found: {path}")
        st.stop()

    df, chosen = _read_excel_all(path)

    # --- HARDEN types ---
    df["ASSESSDATE"] = pd.to_datetime(df["ASSESSDATE"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["ASSESSDATE", "VALUE"])

    df["DESCRIPTION"] = df["DESCRIPTION"].astype(str).fillna("")


    # --- Robust title mapping ---
    def map_title(desc) -> str | None:
        # exact map first
        if desc in TITLE_MAP:
            return TITLE_MAP[desc]

        s = str(desc).lower() if not isinstance(desc, str) else desc.lower()

        if ("resid" in s or "residual" in s) and "ara" in s:
            return "ARA stocks"
        if "padd" in s and "3" in s:
            return "PADD 3 stocks"
        if "fujairah" in s or "fedcom" in s:
            return "Fujairah stocks"
        if "singapore" in s:
            return "Singapore stocks"
        return None

    df["TITLE"] = df["DESCRIPTION"].map(map_title)
    df = df[df["TITLE"].notna()].copy()

    # derive year/week
    df["Year"] = df["ASSESSDATE"].dt.year
    df["Week"] = df["ASSESSDATE"].dt.isocalendar().week.astype(int)

    st.caption(f"Using sheet: **{chosen}** from `{os.path.basename(path)}`")
    return df


# ---------- weekly stats ----------
def weekly_stats(hist_df: pd.DataFrame):
    pivot = hist_df.groupby(["Week", "Year"])["VALUE"].mean().unstack()
    pivot = pivot.reindex(pd.Index(range(1, 54), name="Week"))
    return pivot.mean(axis=1), pivot.min(axis=1), pivot.max(axis=1)


# ---------- change table ----------
def _nearest_on_or_before(series: pd.Series, target_date: pd.Timestamp):
    s = series[series.index <= target_date]
    if s.empty: return None, None
    idx = s.index.max(); return idx, s.loc[idx]

def compute_change_table(df_region: pd.DataFrame):
    ts = df_region.sort_values("ASSESSDATE").set_index("ASSESSDATE")["VALUE"].asfreq("D", method="pad")
    latest_date = ts.dropna().index.max(); latest_val = float(ts.loc[latest_date])

    rows = [["Latest", latest_date.strftime("%d-%m-%Y"), f"{latest_val:,.2f}", "-", "-"]]
    for label, delta in [("Previous week", pd.DateOffset(weeks=1)),
                         ("Previous month", pd.DateOffset(months=1)),
                         ("Previous year", pd.DateOffset(years=1))]:
        d, v = _nearest_on_or_before(ts, latest_date - delta)
        if d is None:
            rows.append([label, "-", "-", "-", "-"])
        else:
            chg = latest_val - v
            pct = (chg / v) * 100 if v else np.nan
            rows.append([label, d.strftime("%d-%m-%Y"), f"{v:,.2f}", f"{chg:,.2f}", f"{pct:,.2f}%" if np.isfinite(pct) else "-"])

    uom = df_region["UOM"].mode().iloc[0] if not df_region["UOM"].empty else ""
    table = pd.DataFrame(rows, columns=["Label", "Date",
                                        f"Value ({uom})" if uom else "Value",
                                        f"Change ({uom})" if uom else "Change",
                                        "Change %"])
    return table, uom


# ---------- Plotly chart ----------
def build_plotly_chart(data: pd.DataFrame, title: str) -> go.Figure:
    hist = data[(data["Year"] >= HIST_START) & (data["Year"] <= HIST_END)]
    mean_vals, min_vals, max_vals = weekly_stats(hist); x = mean_vals.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=min_vals, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=max_vals, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(128,128,128,0.30)", name=f"Range {HIST_START}–{HIST_END}"))
    fig.add_trace(go.Scatter(x=x, y=mean_vals, mode="lines",
                             line=dict(color="black", dash="dash", width=2), name=f"Moyenne {HIST_START}–{HIST_END}"))

    for year, grp in data.groupby("Year"):
        weekly_vals = grp.groupby("Week")["VALUE"].mean().reindex(range(1, 54))
        color, opacity = None, 1.0
        if year == 2025: color = COLOR_2025
        elif year == 2024: color = COLOR_2024
        elif year == 2023: color = COLOR_2023
        else: opacity = OTHER_ALPHA

        fig.add_trace(go.Scatter(x=weekly_vals.index, y=weekly_vals.values,
                                 mode="lines", name=str(year),
                                 line=dict(color=color, width=2) if color else dict(width=2),
                                 opacity=opacity, connectgaps=True))

    fig.update_layout(
        title=f"Stock saisonnier hebdomadaire ({HIST_START}–{dt.date.today().year}) – {title}",
        xaxis_title="Semaine", yaxis_title="Stocks (KT)",
        hovermode="x unified", legend_title="Année",
        margin=dict(l=30, r=10, t=60, b=30), height=520
    )
    fig.update_xaxes(range=[1, 53], dtick=4, showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# ---------- Tab renderer ----------
def generate_stocks_tab():
    st.subheader("Global stocks (interactif)")

    # optional reload button to clear cache when the file updates
    if st.button("🔄 Reload stocks data"):
        _read_excel_all.clear()
        st.rerun()

    df = load_data()
    df = df[df["Year"] >= 2015].copy()

    regions = sorted(df["TITLE"].unique())
    c1, c2 = st.columns([2, 1])
    with c1:
        selected = st.multiselect("Régions", regions, default=regions)
    with c2:
        miny, maxy = int(df["Year"].min()), int(df["Year"].max())
        years = st.slider("Années à afficher", miny, maxy, (max(miny, maxy-5), maxy))

    df = df[df["TITLE"].isin(selected) & df["Year"].between(years[0], years[1])]

    for title in selected:
        temp = df[df["TITLE"] == title]
        st.plotly_chart(build_plotly_chart(temp, title), use_container_width=True)

        table_df, _ = compute_change_table(temp[["ASSESSDATE", "VALUE", "UOM"]])
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"Télécharger la table – {title} (CSV)",
            data=table_df.to_csv(index=False).encode("utf-8"),
            file_name=f"change_table_{title.replace(' ','_')}.csv",
            mime="text/csv",
        )
        st.markdown("---")
