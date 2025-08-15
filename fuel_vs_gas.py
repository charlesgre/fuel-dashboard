# fuel_vs_gas.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def _default_excel_path(repo_root: Path) -> Path:
    return repo_root / "fuel_vs_gas" / "Gas vs Fuel.xlsx"

@st.cache_data(show_spinner=False)
def load_data(excel_path: Path) -> dict:
    df_jkm = pd.read_excel(excel_path, sheet_name="JKM vs 380", skiprows=7, usecols="A,J,K", header=None)
    df_jkm = df_jkm.iloc[1:].reset_index(drop=True)
    df_jkm.columns = ["Date", "JKM", "380_CST"]
    df_jkm["Date"] = pd.to_datetime(df_jkm["Date"], errors="coerce")
    df_jkm["JKM"] = pd.to_numeric(df_jkm["JKM"], errors="coerce")
    df_jkm["380_CST"] = pd.to_numeric(df_jkm["380_CST"], errors="coerce")
    df_jkm = df_jkm.dropna(subset=["Date", "JKM", "380_CST"]).sort_values("Date")

    df_ttf = pd.read_excel(excel_path, sheet_name="TTF vs 1%", skiprows=7, usecols="A,M,Q")
    df_ttf.columns = ["Date", "1pct", "TTF"]
    df_ttf = df_ttf.dropna().copy()
    df_ttf["Date"] = pd.to_datetime(df_ttf["Date"], errors="coerce")
    df_ttf = df_ttf.dropna(subset=["Date"]).sort_values("Date")

    return {"df_jkm": df_jkm, "df_ttf": df_ttf}

def line_2series(df: pd.DataFrame, xcol: str, y1: str, y2: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[xcol], y=df[y1], mode="lines", name=y1))
    fig.add_trace(go.Scatter(x=df[xcol], y=df[y2], mode="lines", name=y2))
    last_x = df[xcol].iloc[-1]
    last_y1 = df[y1].iloc[-1]
    last_y2 = df[y2].iloc[-1]
    fig.add_trace(go.Scatter(x=[last_x], y=[last_y1], mode="markers+text", text=[f"{y1}: {last_y1:.2f}"], textposition="top left", showlegend=False))
    fig.add_trace(go.Scatter(x=[last_x], y=[last_y2], mode="markers+text", text=[f"{y2}: {last_y2:.2f}"], textposition="bottom left", showlegend=False))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="BOE", hovermode="x unified", legend=dict(orientation="h"))
    return fig

def generate_fuel_vs_gas_tab():
    st.header("Gas vs Fuel – Interactive Charts")
    repo_root = Path(__file__).resolve().parent
    excel_candidate = os.getenv("FVG_EXCEL_PATH") or st.secrets.get("FVG_EXCEL_PATH", str(_default_excel_path(repo_root)))
    excel_path = Path(excel_candidate)

    if not excel_path.exists():
        st.error(f"Fichier introuvable : {excel_path}")
        st.stop()

    dfs = load_data(excel_path)
    df_jkm, df_ttf = dfs["df_jkm"], dfs["df_ttf"]

    jkm_cut = df_jkm[df_jkm["Date"] >= (df_jkm["Date"].max() - pd.DateOffset(months=3))]
    ttf_cut = df_ttf[df_ttf["Date"] >= (df_ttf["Date"].max() - pd.DateOffset(months=3))]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(line_2series(df_jkm, "Date", "JKM", "380_CST", "JKM vs 380 CST – Historique"), use_container_width=True)
        st.plotly_chart(line_2series(jkm_cut, "Date", "JKM", "380_CST", "JKM vs 380 CST – 3 mois"), use_container_width=True)
    with c2:
        st.plotly_chart(line_2series(df_ttf, "Date", "TTF", "1pct", "TTF vs 1% Fuel Oil – Historique"), use_container_width=True)
        st.plotly_chart(line_2series(ttf_cut, "Date", "TTF", "1pct", "TTF vs 1% Fuel Oil – 3 mois"), use_container_width=True)