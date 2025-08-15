# fuel_vs_gas.py
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# =========================
# Local path helpers
# =========================
def _pick_local_excel(repo_root: Path) -> Path:
    """
    Retourne le premier chemin existant parmi les variantes locales usuelles.
    Priorité au dossier 'Fuel vs gas' (avec espaces) comme dans ton repo.
    """
    candidates = [
        repo_root / "Fuel vs gas" / "Gas vs Fuel.xlsx",   # ← ton dossier actuel
        repo_root / "Fuel vs Gas" / "Gas vs Fuel.xlsx",   # variante de casse
        repo_root / "fuel_vs_gas" / "Gas vs Fuel.xlsx",   # variante underscores
        repo_root / "Gas vs Fuel.xlsx",                   # racine du repo
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # valeur par défaut (même si non existant)


# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_data_from_path(excel_path: Path) -> dict:
    """
    Charge les deux onglets utiles depuis l'Excel et renvoie
    {'df_jkm': DataFrame, 'df_ttf': DataFrame}
    """
    # --- JKM vs 380 ---
    df_jkm = pd.read_excel(
        excel_path, sheet_name="JKM vs 380", skiprows=7, usecols="A,J,K", header=None
    )
    df_jkm = df_jkm.iloc[1:].reset_index(drop=True)
    df_jkm.columns = ["Date", "JKM", "380_CST"]
    df_jkm["Date"] = pd.to_datetime(df_jkm["Date"], errors="coerce")
    df_jkm["JKM"] = pd.to_numeric(df_jkm["JKM"], errors="coerce")
    df_jkm["380_CST"] = pd.to_numeric(df_jkm["380_CST"], errors="coerce")
    df_jkm = df_jkm.dropna(subset=["Date", "JKM", "380_CST"]).sort_values("Date")

    # --- TTF vs 1% ---
    df_ttf = pd.read_excel(
        excel_path, sheet_name="TTF vs 1%", skiprows=7, usecols="A,M,Q"
    )
    df_ttf.columns = ["Date", "1pct", "TTF"]
    df_ttf = df_ttf.dropna().copy()
    df_ttf["Date"] = pd.to_datetime(df_ttf["Date"], errors="coerce")
    df_ttf = df_ttf.dropna(subset=["Date"]).sort_values("Date")

    return {"df_jkm": df_jkm, "df_ttf": df_ttf}


# =========================
# Plotly chart helper
# =========================
def line_2series(df: pd.DataFrame, xcol: str, y1: str, y2: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[xcol], y=df[y1], mode="lines", name=y1))
    fig.add_trace(go.Scatter(x=df[xcol], y=df[y2], mode="lines", name=y2))

    # Marqueurs + labels sur la dernière observation
    last_x = df[xcol].iloc[-1]
    last_y1 = df[y1].iloc[-1]
    last_y2 = df[y2].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[last_x],
            y=[last_y1],
            mode="markers+text",
            text=[f"{y1}: {last_y1:.2f}"],
            textposition="top left",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_x],
            y=[last_y2],
            mode="markers+text",
            text=[f"{y2}: {last_y2:.2f}"],
            textposition="bottom left",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="BOE (Barrel of Oil Eq.)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# =========================
# Streamlit tab
# =========================
def generate_fuel_vs_gas_tab():
    st.header("Gas vs Fuel – Interactive Charts")

    repo_root = Path(__file__).resolve().parent

    # 1) secrets/env si fournis, sinon 2) chemin local auto (variante 'Fuel vs gas/...')
    excel_candidate = os.getenv("FVG_EXCEL_PATH") or st.secrets.get("FVG_EXCEL_PATH")
    excel_path = Path(excel_candidate) if excel_candidate else _pick_local_excel(repo_root)

    # Si le fichier n'existe pas, permettre l'upload manuel, sans bloquer l'app
    dfs = None
    with st.expander("Source de données", expanded=not excel_path.exists()):
        st.caption("Chemin détecté : {}".format(excel_path))
        if not excel_path.exists():
            up = st.file_uploader("Importer l'Excel (Gas vs Fuel.xlsx)", type=["xlsx"])
            if up is not None:
                dfs = load_data_from_path(Path(up.name))  # lecture via buffer temp
                # Astuce: pandas a besoin d’un chemin/bytes; on lit directement depuis l'uploader
                # Re-lecture correcte:
                up.seek(0)
                dfs = {
                    "df_jkm": pd.read_excel(up, sheet_name="JKM vs 380", skiprows=7, usecols="A,J,K", header=None)
                        .iloc[1:].rename(columns={0:"Date",1:"JKM",2:"380_CST"}),
                    "df_ttf": pd.read_excel(up, sheet_name="TTF vs 1%",  skiprows=7, usecols="A,M,Q")
                        .rename(columns={"A":"Date","M":"1pct","Q":"TTF"}),
                }
                # Nettoyage minimal identique
                for key, df in dfs.items():
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    for c in df.columns:
                        if c != "Date":
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    dfs[key] = df.dropna(subset=["Date"]).sort_values("Date")
            else:
                st.info("Place l'Excel dans `Fuel vs gas/Gas vs Fuel.xlsx` ou configure `FVG_EXCEL_PATH`.")
        else:
            dfs = load_data_from_path(excel_path)

    if dfs is None:
        return  # on sort proprement si aucune donnée

    df_jkm = dfs["df_jkm"].copy()
    df_ttf = dfs["df_ttf"].copy()

    # Filtres de période rapides
    preset = st.radio(
        "Période",
        ["3M", "YTD", "Depuis 2022", "Tout"],
        horizontal=True,
        index=0,
    )

    def cut(df: pd.DataFrame):
        if preset == "3M":
            start = df["Date"].max() - pd.DateOffset(months=3)
        elif preset == "YTD":
            start = pd.Timestamp(pd.Timestamp.today().year, 1, 1)
        elif preset == "Depuis 2022":
            start = pd.Timestamp("2022-01-01")
        else:
            start = df["Date"].min()
        return df[df["Date"] >= start]

    jkm_cut = cut(df_jkm)
    ttf_cut = cut(df_ttf)

    # Affichage (2 colonnes, 2 graphs chacune : historique sélectionné)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("JKM vs 380 CST")
        st.plotly_chart(
            line_2series(jkm_cut, "Date", "JKM", "380_CST",
                         f"JKM vs 380 CST – {preset}"),
            use_container_width=True,
            key="jkm_hist",
        )

    with c2:
        st.subheader("TTF vs 1% Fuel Oil")
        st.plotly_chart(
            line_2series(ttf_cut, "Date", "TTF", "1pct",
                         f"TTF vs 1% Fuel Oil – {preset}"),
            use_container_width=True,
            key="ttf_hist",
        )
