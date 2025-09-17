# balances/litasco_supply_tab.py
# -*- coding: utf-8 -*-
import os
import platform
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------------- Dossier local (type "Bunker diff/…") ----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LITASCO_DIR = REPO_ROOT / "Litasco balances" / "Litasco supply"
LOCAL_LITASCO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- Paramétrage régions ----------------------
REFINERIES_BY_REGION = {
    "NWE": {
        "Belgium": [
            {"Refinery": "Antwerpen", "Capacity": 320, "Yields": {"VLSFO": 0.10}},
            {"Refinery": "Antwerp",   "Capacity": 362, "Yields": {"HSFO": 0.06, "VLSFO": 0.06}},
        ],
        "United Kingdom": [
            {"Refinery": "Stanlow",      "Capacity": 205, "Yields": {"LSFO": 0.03}},
            {"Refinery": "Pembroke",     "Capacity": 210, "Yields": {"LSFO": 0.05, "VGO": 0.02}},
            {"Refinery": "Lindsey",      "Capacity": 113, "Yields": {"VGO": 0.04}},
            {"Refinery": "Grangemouth",  "Capacity": 150, "Yields": {"LSSR": 0.04}},
            {"Refinery": "Eastham",      "Capacity": 20,  "Yields": {"HSSR": 0.11}},
        ],
        "Sweden": [
            {"Refinery": "Preemraff Gothenburg", "Capacity": 126, "Yields": {"VGO": 0.03, "LSSR": 0.17}},
            {"Refinery": "St1 Gothenburg A B",   "Capacity": 83,  "Yields": {"VGO": 0.01, "LSSR": 0.05}},
        ],
        "France": [
            {"Refinery": "Donges",                 "Capacity": 230, "Yields": {"VLSFO": 0.07}},
            {"Refinery": "Port-Jerome Gravenchon", "Capacity": 240, "Yields": {"HSFO": 0.01, "VGO": 0.01}},
        ],
        "Netherlands": [
            {"Refinery": "Zeeland",       "Capacity": 155, "Yields": {"VLSFO": 0.04}},
            {"Refinery": "Pernis",        "Capacity": 400, "Yields": {"VLSFO": 0.11}},
            {"Refinery": "BP Rotterdam",  "Capacity": 400, "Yields": {"VLSFO": 0.13}},
        ],
        "Denmark": [
            {"Refinery": "Frederica",     "Capacity": 75,  "Yields": {"VLSFO": 0.13}},
            {"Refinery": "Kalundborg",    "Capacity": 105, "Yields": {"LSFO": 0.12}},
        ],
        "Finland": [
            {"Refinery": "Porvoo",        "Capacity": 96,  "Yields": {"VGO": 0.03}},
        ],
        "Ireland": [
            {"Refinery": "Whitegate",     "Capacity": 75,  "Yields": {"LSSR": 0.16}},
        ],
        "Lithuania": [
            {"Refinery": "Mazeikiai",     "Capacity": 200, "Yields": {"HSFO": 0.08}},
        ],
        "Poland": [
            {"Refinery": "Gdansk",        "Capacity": 210, "Yields": {"HSFO": 0.02}},
            {"Refinery": "Plock",         "Capacity": 360, "Yields": {"HSFO": 0.02}},
        ],
    },
    "MED": {
        # 👉 prêt à remplir plus tard (même structure que NWE)
    }
}

# ---------------------- Constantes & style ----------------------
KBD_TO_KT = 6.35
BAND_START, BAND_END = 2020, 2024
SPECIAL_COLORS = {2026: "red", 2025: "black", 2024: "green"}
MONTH_TICKS = list(range(1, 13))
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------- IO helpers ----------------------
DATE_IN_NAME = re.compile(r"(\d{2}\.\d{2}\.\d{4})")  # ex: 09.16.2025

def _parse_date_from_name(p: Path):
    m = DATE_IN_NAME.search(p.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%m.%d.%Y")
    except Exception:
        return None

def pick_runs_path(explicit_path: str, pattern: str) -> str:
    """
    Priorité :
      1) chemin explicite si fourni,
      2) sinon dernier fichier LOCAL dans 'Litasco balances/Litasco supply'
         qui matche `pattern` (tri d'abord par date dans le nom, sinon par mtime).
    """
    # 1) chemin explicite
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p)

    # 2) recherche locale
    files = sorted(LOCAL_LITASCO_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            "Aucun fichier 'Runs' trouvé en local.\n"
            f"Place un fichier dans : {LOCAL_LITASCO_DIR}\n"
            f"Pattern utilisé : {pattern}\n"
            "💡 Tu peux aussi déposer le fichier via l’uploader ci-dessus."
        )

    dated = [(f, _parse_date_from_name(f)) for f in files]
    if any(d is not None for _, d in dated):
        files = [f for f, _ in sorted(dated, key=lambda x: (x[1] is None, x[1]), reverse=True)]
    else:
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return str(files[0])

def load_runs(runs_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(runs_path, sheet_name=sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")

def series_for_country(runs_df: pd.DataFrame, country: str) -> pd.Series | None:
    if country in runs_df.columns:
        return runs_df[country]
    if country == "United Kingdom" and "UK" in runs_df.columns:
        return runs_df["UK"]
    return None

def nonempty_series(s: pd.Series | None) -> bool:
    if s is None:
        return False
    vals = pd.to_numeric(s, errors="coerce")
    return not (vals.isna().all() or np.nan_to_num(vals.values).sum() == 0.0)

# ---------------------- Plotly (interactif) ----------------------
def seasonality_figure(series: pd.Series, title: str) -> go.Figure | None:
    s = series.dropna()
    if s.empty or (s == 0).all():
        return None

    df = pd.DataFrame({"kt": s})
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    ym = (df.groupby(["Year","Month"])["kt"].mean()
            .unstack(level=0)
            .reindex(index=MONTH_TICKS))

    years = [int(y) for y in ym.columns if pd.notna(y)]
    if not years:
        return None

    band_years = [y for y in range(BAND_START, BAND_END + 1) if y in years]
    band_min = band_max = None
    if band_years:
        band_data = ym[band_years]
        band_min = band_data.min(axis=1)
        band_max = band_data.max(axis=1)

    fig = go.Figure()

    if band_years and band_min.notna().any() and band_max.notna().any():
        fig.add_trace(go.Scatter(x=MONTH_TICKS, y=band_min.values,
                                 mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=MONTH_TICKS, y=band_max.values,
                                 mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.20)",
                                 name=f"{BAND_START}–{BAND_END} range", hoverinfo="skip"))

    for y in sorted(years):
        line = ym[y]
        color = SPECIAL_COLORS.get(y, None)
        fig.add_trace(go.Scatter(
            x=MONTH_TICKS, y=line.values, mode="lines+markers", name=str(y),
            line=dict(width=2 if y in SPECIAL_COLORS else 1.3, color=color),
            opacity=1.0 if y in SPECIAL_COLORS else 0.6
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=MONTH_TICKS, ticktext=MONTH_LABELS),
        yaxis_title="kt",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420
    )
    return fig

# ---------------------- MAIN TAB RENDERER ----------------------
def run_litasco_supply_tab():
    st.subheader("Litasco supply — seasonality (interactive)")

    c1, c2 = st.columns([1,1])
    with c1:
        region = st.selectbox("Région", ["NWE", "MED"], index=0, help="MED prêt pour plus tard")
    with c2:
        runs_sheet = st.text_input("Onglet Excel (sheet)", value="NWE" if region == "NWE" else "MED")

    # Chemins d'entrée (LOCAL au repo)
    runs_file_explicit = st.text_input(
        "Fichier Runs (chemin complet facultatif — sinon on prend le plus récent en LOCAL)",
        value=""
    )
    runs_pattern = st.text_input(
        "Pattern de fichier (local au repo)",
        value="Europe Runs Recap*.xlsx"
    )
    st.caption(f"Dossier local utilisé : {LOCAL_LITASCO_DIR}")

    # Uploader : dépose le .xlsx ici (il sera sauvegardé dans le dossier local ci-dessus)
    uploaded = st.file_uploader("…ou dépose un fichier .xlsx ici", type=["xlsx"])
    if uploaded is not None:
        save_path = LOCAL_LITASCO_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        runs_file_explicit = str(save_path)
        st.success(f"Fichier uploadé : {save_path.name}")

    st.markdown("---")
    btn_generate = st.button("Générer les graphiques")
    if not btn_generate:
        st.info("Choisis ta région et tes fichiers, puis clique **Générer les graphiques**.")
        return

    # Chargement runs
    try:
        runs_path = pick_runs_path(runs_file_explicit, runs_pattern)
        st.success(f"Fichier Runs utilisé : **{os.path.basename(runs_path)}**")
        st.caption(runs_path)
    except Exception as e:
        st.error(f"Erreur Runs : {e}")
        return

    try:
        runs_df = load_runs(runs_path, runs_sheet)
    except Exception as e:
        st.error(f"Lecture Excel : {e}")
        return

    REFINERIES = REFINERIES_BY_REGION.get(region, {})
    if not REFINERIES:
        st.warning(f"Aucune raffinerie configurée pour {region} (à compléter).")
        return

    target_countries = list(REFINERIES.keys())
    all_products = sorted({prod for plist in REFINERIES.values() for r in plist for prod in r["Yields"].keys()})

    refinery_product: dict[tuple[str, str], pd.DataFrame] = {}
    country_product: dict[str, dict[str, pd.Series]] = {c: {} for c in target_countries}
    totals_product: dict[str, pd.Series | None] = {p: None for p in all_products}

    # Calculs
    for country in target_countries:
        runs_c = series_for_country(runs_df, country)
        if not nonempty_series(runs_c):
            continue

        refs = REFINERIES[country]
        total_cap = sum(r["Capacity"] for r in refs)
        if total_cap == 0:
            continue

        util = runs_c / total_cap  # utilisation

        for p in all_products:
            country_product[country][p] = None

        for r in refs:
            Ri = r["Capacity"] * util  # kbd
            prod_cols = {}
            for prod, y in r["Yields"].items():
                series_kt = Ri * y * KBD_TO_KT  # kt
                series_kt.name = prod
                prod_cols[prod] = series_kt

                country_product[country][prod] = (
                    series_kt.copy() if country_product[country][prod] is None
                    else country_product[country][prod].add(series_kt, fill_value=0.0)
                )
                totals_product[prod] = (
                    series_kt.copy() if totals_product[prod] is None
                    else totals_product[prod].add(series_kt, fill_value=0.0)
                )

            for p in all_products:
                if p not in prod_cols:
                    prod_cols[p] = pd.Series(0.0, index=Ri.index, name=p)

            df_ref = pd.DataFrame(prod_cols, index=Ri.index)[all_products]
            refinery_product[(country, r["Refinery"])] = df_ref

    st.success("Graphiques calculés. Affichage…")

    # AFFICHAGE : pays -> raffineries -> totaux pays
    for country in sorted(target_countries):
        has_any = any((country, ref["Refinery"]) in refinery_product for ref in REFINERIES.get(country, []))
        if not has_any:
            continue

        st.header(country)

        with st.expander("Refineries", expanded=False):
            for ref in REFINERIES[country]:
                key = (country, ref["Refinery"])
                if key not in refinery_product:
                    continue
                df_ref = refinery_product[key]
                st.subheader(f"{ref['Refinery']}")
                cols = st.columns(3)
                col_idx = 0
                for product in df_ref.columns:
                    s = df_ref[product]
                    if s.isna().all() or (s == 0).all():
                        continue
                    fig = seasonality_figure(s, f"{country} / {ref['Refinery']} / {product}")
                    if fig:
                        with cols[col_idx]:
                            st.plotly_chart(fig, use_container_width=True)
                        col_idx = (col_idx + 1) % 3

        st.subheader("Country totals")
        cols = st.columns(3)
        col_idx = 0
        for product, s in country_product[country].items():
            if s is None or s.isna().all() or (np.nan_to_num(s.values).sum() == 0.0):
                continue
            fig = seasonality_figure(s, f"{country} / {product}")
            if fig:
                with cols[col_idx]:
                    st.plotly_chart(fig, use_container_width=True)
                col_idx = (col_idx + 1) % 3

        st.markdown("---")

    # Totaux régionaux
    st.header(f"{region} — Totaux régionaux")
    cols = st.columns(3)
    col_idx = 0
    for product, s in sorted(totals_product.items(), key=lambda kv: kv[0].lower()):
        if s is None or s.isna().all() or (np.nan_to_num(s.values).sum() == 0.0):
            continue
        fig = seasonality_figure(s, f"{region} Total / {product}")
        if fig:
            with cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True)
            col_idx = (col_idx + 1) % 3

    # Résumé CSV
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for country, prod_map in country_product.items():
        for product, s in prod_map.items():
            if s is None:
                continue
            rows.append({
                "Region": region, "Country": country, "Product": product,
                "Avg_kt": round(float(pd.to_numeric(s, errors="coerce").mean()), 2),
                "Total_kt": round(float(pd.to_numeric(s, errors="coerce").sum()), 2),
            })
    for product, s in totals_product.items():
        if s is None:
            continue
        rows.append({
            "Region": region, "Country": f"{region} Total", "Product": product,
            "Avg_kt": round(float(pd.to_numeric(s, errors="coerce").mean()), 2),
            "Total_kt": round(float(pd.to_numeric(s, errors="coerce").sum()), 2),
        })

    if rows:
        df_summary = pd.DataFrame(rows).sort_values(["Region","Country","Product"]).reset_index(drop=True)
        st.subheader("Résumé (tableau)")
        st.dataframe(df_summary, use_container_width=True)
        csv_bytes = df_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("Télécharger le CSV résumé",
                           data=csv_bytes,
                           file_name=f"Summary_{region}_{today}.csv",
                           mime="text/csv")
    else:
        st.info("Aucune donnée non-nulle à résumer.")
