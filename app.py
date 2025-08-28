# app.py
import os
import platform
from pathlib import Path
from datetime import datetime

import streamlit as st

from generate_charts import generate_price_charts
from bunker_diff import plot_bunker_price_diffs
from fge_balances import plot_fge_balances, load_fge_balances
from forward_curves import generate_forward_curves_tab
from forward_curves_us import generate_us_forward_curves_tab
from streamlit_platts_tab import generate_platts_analytics_tab
from generate_stocks_tab import generate_stocks_tab
from fuel_vs_gas import generate_fuel_vs_gas_tab
from technical_analysis_tab import render as render_technical
from arbs_tab import render as render_arbs
from sg_imports_tab import render_sg_imports_tab

# ---- Secrets / .env helpers (robust even if secrets.toml isn't auto-loaded) ----
from typing import Optional
from pathlib import Path

# (optional) load .env if you use one locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _load_project_secrets_into_env() -> None:
    """Load .streamlit/secrets.toml if present and push to os.environ."""
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return
    text = secrets_path.read_text(encoding="utf-8")

    # 1) Try Python 3.11+ tomllib
    data = None
    try:
        import tomllib  # type: ignore
        data = tomllib.loads(text)
    except Exception:
        # 2) Try 'toml' package
        try:
            import toml  # type: ignore
            data = toml.loads(text)
        except Exception:
            # 3) Ultra-simple fallback parser (clé = "valeur" en 1er niveau)
            data = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v:
                    data[k] = v

    for k, v in (data or {}).items():
        if isinstance(v, (str, int, float)) and not os.getenv(k):
            os.environ[k] = str(v)

# Load local secrets file into env (if present)
_load_project_secrets_into_env()

def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Prefer environment variable; only touch st.secrets if present
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]  # raises if secrets.toml is missing
    except Exception:
        return default

# Sync Streamlit secrets -> environment variables
for k in ("PL_USERNAME", "PL_PASSWORD", "PL_API_KEY", "PL_API_HASH", "EA_PDF_DIR"):
    v = _get_secret(k, None)
    if v and not os.getenv(k):
        os.environ[k] = v

from key_locations import render_key_locations_export



# ------------ Page config ------------
st.set_page_config(page_title="Fuel Dashboard", layout="wide")
st.title("📊 Fuel Dashboard")

if st.button("🔄 Force refresh"):
    st.cache_data.clear()
    st.rerun()

# ------------ Config EA_PDF_DIR (AVANT d'importer ea_balances) ------------
repo_root = Path(__file__).resolve().parent
os.environ["FUEL_DASH_DATA_ROOT"] = str(repo_root)
local_default = repo_root / "EA balances"

EA_DIR = _get_secret("EA_PDF_DIR", str(local_default))
if platform.system() != "Windows" and (EA_DIR.startswith("\\") or EA_DIR.startswith("//")):
    EA_DIR = str(local_default)
os.environ["EA_PDF_DIR"] = EA_DIR

# ⚠️ Import APRÈS config du path
from ea_balances import (  # noqa: E402
    load_ea_data as _load_ea_data,
    plot_ea,
    PARSER_VERSION,
    _get_latest_pdf_file as pick_ea_pdf,
)

# ------------ Cache EA dépendant de la version du parseur ------------
@st.cache_data(show_spinner=False)
def get_ea_data_cached(_parser_version: str):
    return _load_ea_data()

# ------------ Tabs ------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "📊 Prices", "⛽ Bunker Diff", "CDD/Temperatures",
    "Balances (FGE / EA)", "📈 Forward Curves", "Platts Window",
    "📦 Fuel Stocks", "Gas vs Fuel", "🔍 Technical Analysis", "📑 Arbs",
    "🇸🇬 SG hub tracking",  # 👈 NOUVEL ONGLET
    "🌍 Key locations export"
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
    try:
        from cdd_temperatures import get_all_cdd_figures
        figures = get_all_cdd_figures()
    except Exception as e:
        st.error("Erreur CDD / Temperatures")
        with st.expander("Traceback complet"):
            st.exception(e)
        st.stop()

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
        with st.expander("EA – PDF utilisé (debug)", expanded=False):
            st.caption(f"Dossier EA_PDF_DIR: {EA_DIR}")
            if platform.system() != "Windows" and (EA_DIR.startswith('\\') or EA_DIR.startswith('//')):
                st.warning(
                    "Chemin UNC détecté sur un runtime Linux : non accessible directement.\n"
                    "➡️ Copie le PDF dans le repo 'EA balances' ou monte le partage réseau."
                )
            try:
                pdf_path = pick_ea_pdf()
                st.info(f"PDF choisi par le parseur : **{pdf_path.name}**")
            except Exception as e:
                st.warning(f"Impossible d’évaluer le PDF choisi : {e}")

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

# === TAB 7: STOCKS ===  ✅ protégée contre FileNotFoundError
with tab7:
    st.header("📦 Fuel Stocks – Seasonal Charts & Comparisons")
    try:
        generate_stocks_tab()
    except FileNotFoundError as e:
        st.error("Le fichier Excel des stocks est introuvable.")
        with st.expander("Détails"):
            st.exception(e)
    except Exception as e:
        st.error("Erreur dans l’onglet Stocks.")
        with st.expander("Traceback complet"):
            st.exception(e)

# === TAB 8: GAS vs FUEL ===
with tab8:
    generate_fuel_vs_gas_tab()

# === TAB 9: TECHNICAL ANALYSIS ===
with tab9:
    render_technical()

# === TAB 10: ARBS ===
with tab10:
    render_arbs()

# === TAB 11: SG imports ===
with tab11:
    render_sg_imports_tab()   # ne pas passer default_dir

# === TAB 12: KEY LOCATIONS EXPORT ===
with tab12:
    def _mask(v: str):
        if not v:
            return "❌ missing"
        return "✅ set (" + ("*" * max(0, len(v) - 4) + v[-4:]) + ")"


    st.caption("Petro-Logistics credentials status:")
    st.write({
        "PL_USERNAME": _mask(os.getenv("PL_USERNAME", "")),
        "PL_PASSWORD": _mask(os.getenv("PL_PASSWORD", "")),
        "PL_API_KEY":  _mask(os.getenv("PL_API_KEY", "")),
        "PL_API_HASH": _mask(os.getenv("PL_API_HASH", "")),
    })

    render_key_locations_export()