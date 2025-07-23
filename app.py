import streamlit as st
from generate_charts import generate_price_charts
from bunker_diff import plot_bunker_price_diffs
from cdd_temperatures import get_all_cdd_figures
from fge_balances import plot_fge_balances, load_fge_balances
from forward_curves import generate_forward_curves_tab
from forward_curves_us import generate_us_forward_curves_tab
from streamlit_platts_tab import generate_platts_analytics_tab
from generate_stocks_tab import generate_stocks_tab  # ✅ nouvelle tab importée

from datetime import datetime

st.set_page_config(page_title="Fuel Dashboard", layout="wide")
st.title("📊 Fuel Dashboard")

# ✅ Ajout d'une 7ème tab
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Prices", "⛽ Bunker Diff", "CDD/Temperatures",
    "FGE balances", "📈 Forward Curves", "Platts Window", "📦 Fuel Stocks"
])

# === TAB 1: PRICES ===
with tab1:
    st.header("Seasonality Charts – Main Benchmarks")

    all_titles = list(generate_price_charts().keys())
    charts = generate_price_charts(all_titles)

    cols = st.columns(3)
    col_idx = 0

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
    st.write(f"Figures CDD récupérées ({len(figures)}): {list(figures.keys())}")  # debug affichage

    # Egypt
    st.subheader("Egypt")
    egypt_figs = {k: v for k, v in figures.items() if 'Egypt' in k}
    cols = st.columns(3)
    col_idx = 0
    for i, (title, fig) in enumerate(egypt_figs.items()):
        with cols[col_idx]:
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=True, key=f"egypt_cdd_{i}")
        col_idx = (col_idx + 1) % 3

    st.markdown("---")

    # Saudi Arabia
    st.subheader("Saudi Arabia")
    saudi_figs = {k: v for k, v in figures.items() if 'Saudi' in k}
    cols = st.columns(3)
    col_idx = 0
    for i, (title, fig) in enumerate(saudi_figs.items()):
        with cols[col_idx]:
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=True, key=f"saudi_cdd_{i}")
        col_idx = (col_idx + 1) % 3

# === TAB 4: FGE BALANCES ===
with tab4:
    st.header("FGE Seasonal Balances")

    vlsfo_data, hsfo_data = load_fge_balances()

    st.subheader("VLSFO")
    vlsfo_figs = plot_fge_balances(vlsfo_data, "VLSFO")
    cols = st.columns(3)
    col_idx = 0
    for i, (title, fig) in enumerate(vlsfo_figs.items()):
        with cols[col_idx]:
            st.plotly_chart(fig, use_container_width=True)
        col_idx = (col_idx + 1) % 3

    st.markdown("---")

    st.subheader("HSFO")
    hsfo_figs = plot_fge_balances(hsfo_data, "HSFO")
    cols = st.columns(3)
    col_idx = 0
    for i, (title, fig) in enumerate(hsfo_figs.items()):
        with cols[col_idx]:
            st.plotly_chart(fig, use_container_width=True)
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

# === TAB 7: STOCKS === ✅ NOUVELLE TAB
with tab7:
    st.header("📦 Fuel Stocks – Seasonal Charts & Comparisons")
    generate_stocks_tab()
