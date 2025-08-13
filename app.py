# app.py
import os
import platform
from pathlib import Path
from datetime import datetime

import streamlit as st

from generate_charts import generate_price_charts
from bunker_diff import plot_bunker_price_diffs
from cdd_temperatures import get_all_cdd_figures
from fge_balances import plot_fge_balances, load_fge_balances
from forward_curves import generate_forward_curves_tab
from forward_curves_us import generate_us_forward_curves_tab
from streamlit_platts_tab import generate_platts_analytics_tab
from generate_stocks_tab import generate_stocks_tab

# ------------ Page config ------------
st.set_page_config(page_title="Fuel Dashboard", layout="wide")
st.title("📊 Fuel Dashboard")

# ------------ Config EA_PDF_DIR (doit être fait AVANT d'importer ea_balances) ------------
repo_root = Path(__file__).resolve().parent
local_default = repo_root / "EA balances"   # dossier DU REPO (doit exister en prod)

# ordre: env -> secrets -> fallback local
EA_DIR = os.getenv("EA_PDF_DIR", st.secrets.get("EA_PDF_DIR", str(local_default)))

# ⛑️ Protection: si on n'est PAS sous Windows et que le chemin ressemble à un UNC, basculer sur le fallback local
if platform.system() != "Windows" and (EA_DIR.startswith("\\") or EA_DIR.startswith("//")):
    EA_DIR = str(local_default)

# Expose au runtime pour le parseur
os.environ["EA_PDF_DIR"] = EA_DIR
st.caption(f"EA_PDF_DIR utilisé: {EA_DIR}")

# Debug rapide pour voir ce que le runtime voit
with st.expander("EA debug"):
    p = Path(EA_DIR)
    st.write("OS:", platform.system(), "| CWD:", Path.cwd())
    st.write("Existe:", p.exists())
    try:
        st.write("Fichiers (extrait):", [x.name for x in p.glob("*.pdf")][:10])
    except Exception as e:
        st.write("Listing impossible:", e)

# ⚠️ Import APRÈS config du path
from ea_balances import (  # noqa: E402
    load_ea_data as _load_ea_data,
    plot_ea,
    PARSER_VERSION,   # utilisé pour clé de cache
)

# ------------ Cache EA dépendant de la version du parseur ------------
@st.cache_data(show_spinner=False)
def get_ea_data_cached(_parser_version: str):
    """
    Le paramètre _parser_version est uniquement là pour 'cléer' le cache.
    Dès que PARSER_VERSION change dans ea_balances.py, le cache est invalidé.
    """
    return _load_ea_data()

# ------------ Tabs ------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Prices", "⛽ Bunker Diff", "CDD/Temperatures",
    "Balances (FGE / EA)", "📈 Forward Curves", "Platts Window", "📦 Fuel Stocks"
])

# === TAB 1: PRICES ===
with tab1:
    st.header("Seasonality Charts – Main Benchmarks")
    all_titles = list(generate_price_charts().keys())
    charts = generate_price_charts(all_titles)
    cols = st.columns(3); col_idx = 0
    for i, (title, fig) in enumerate(charts.items()):
        with cols[col_idx]:
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=True, key=f"price_{i}")
        col_idx = (col_idx + 1) % 3

# === TAB 2: BUNKER DIFF ===
with tab2:
    st.header("Bunker Price Differentials")
    plot_bunker_price_diffs()

# === TAB 3: CDD / TEMPERATURES ===
with tab3:
    st.header("CDD / Temperatures")
    figures = get_all_cdd_figures()
    st.write(f"Figures CDD récupérées ({len(figures)}): {list(figures.keys())}")
    st.subheader("Egypt")
    egypt_figs = {k: v for k, v in figures.items() if 'Egypt' in k}
    cols = st.columns(3); col_idx = 0
    for i, (title, fig) in enumerate(egypt_figs.items()):
        with cols[col_idx]:
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=True, key=f"egypt_cdd_{i}")
        col_idx = (col_idx + 1) % 3
    st.markdown("---")
    st.subheader("Saudi Arabia")
    saudi_figs = {k: v for k, v in figures.items() if 'Saudi' in k}
    cols = st.columns(3); col_idx = 0
    for i, (title, fig) in enumerate(saudi_figs.items()):
        with cols[col_idx]:
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=True, key=f"saudi_cdd_{i}")
        col_idx = (col_idx + 1) % 3

# === TAB 4: BALANCES (FGE / EA) ===
with tab4:
    st.header("Seasonal Balances – FGE & EA")
    source = st.radio("Source de données", ["FGE", "EA"], index=0, horizontal=True)

    if source == "FGE":
        vlsfo_data, hsfo_data = load_fge_balances()

        st.subheader("VLSFO (FGE)")
        vlsfo_figs = plot_fge_balances(vlsfo_data, "VLSFO")
        cols = st.columns(3); col_idx = 0
        for i, (title, fig) in enumerate(vlsfo_figs.items()):
            with cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True, key=f"fge_vlsfo_{i}")
            col_idx = (col_idx + 1) % 3

        st.markdown("---")

        st.subheader("HSFO (FGE)")
        hsfo_figs = plot_fge_balances(hsfo_data, "HSFO")
        cols = st.columns(3); col_idx = 0
        for i, (title, fig) in enumerate(hsfo_figs.items()):
            with cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True, key=f"fge_hsfo_{i}")
            col_idx = (col_idx + 1) % 3

    else:
        st.subheader("EA (Europe fuel oil – Fig.10)")
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            metric = st.selectbox("Metric", ["Balance", "Demand", "Supply"], index=0)
        with c2:
            grade = st.radio("Grade", ["HSFO", "LSFO"], index=0, horizontal=True)
        with c3:
            if st.button("🔄 Reparser EA (clear cache)"):
                get_ea_data_cached.clear()
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        with st.spinner("Chargement EA…"):
            try:
                # 👇 le cache dépend de la version du parseur
                ea_data = get_ea_data_cached(PARSER_VERSION)
            except FileNotFoundError as e:
                st.error(f"EA_PDF_DIR: {EA_DIR}\n{e}")
                st.stop()
            except Exception as e:
                st.exception(e)
                st.stop()

        figs = plot_ea(ea_data, metric=metric, grade=grade)
        cols = st.columns(3); col_idx = 0
        for i, (title, fig) in enumerate(figs.items()):
            with cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True,
                                key=f"ea_{metric}_{grade}_{i}")
            col_idx = (col_idx + 1) % 3

# === TAB 5: FORWARD CURVES ===
with tab5:
    st.header("📈 Forward Curves")
    st.subheader("🇪🇺 ARA / Singapore Forward Curves")
    generate_forward_curves_tab()
    st.markdown("---")
    st.subheader("🇺🇸 US Forward Curves")
    generate_us_forward_curves_tab()

# === TAB 6: PLATTS ===
with tab6:
    st.header("Platts Window Analytics")
    generate_platts_analytics_tab()

# === TAB 7: STOCKS ===
with tab7:
    st.header("📦 Fuel Stocks – Seasonal Charts & Comparisons")
    generate_stocks_tab()
