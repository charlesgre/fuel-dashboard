# balances/litasco_supply_tab.py
# -*- coding: utf-8 -*-
import os, glob, platform
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# --------- Dossier par défaut pour les fichiers Runs (UNC) ---------
DEFAULT_LITASCO_RUNS_DIR = r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\Litasco balances\Litasco supply"

# Fallback local (comme "Bunker diff/…") pour les environnements non-Windows
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LITASCO_DIR = REPO_ROOT / "Litasco balances" / "Litasco supply"
LOCAL_LITASCO_DIR.mkdir(parents=True, exist_ok=True)  # ok si déjà présent

# ========== PARAMETRISATION REGION ==========
# Dictionnaires par région. NWE est rempli; MED est un placeholder prêt à être complété.
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
        # 👉 à compléter plus tard (capacités & rendements) au même format que NWE.
    }
}

# --------- Constantes / style ---------
KBD_TO_KT = 6.35
BAND_START, BAND_END = 2020, 2024
SPECIAL_COLORS = {2026: "red", 2025: "black", 2024: "green"}
MONTH_TICKS = list(range(1, 13))
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ================== IO HELPERS ==================
def pick_runs_path(explicit_path: str, runs_dir: str, pattern: str) -> str:
    """
    1) Si explicit_path pointe vers un fichier existant -> return.
    2) Sinon, on tente le dossier UNC (Windows uniquement).
    3) Sinon, on tente le dossier local du repo: 'Litasco balances/Litasco supply'.
    Retourne le fichier le plus récent qui matche pattern.
    """
    # 1) Chemin explicite
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    candidates = []

    # 2) Dossier UNC (seulement si Windows)
    if platform.system() == "Windows":
        unc_dir = runs_dir.replace("\\", "/")
        candidates.append(os.path.join(unc_dir, pattern))

    # 3) Fallback local relatif au repo
    local_dir = str(LOCAL_LITASCO_DIR).replace("\\", "/")
    candidates.append(os.path.join(local_dir, pattern))

    # Recherche
    matched = []
    for pat in candidates:
        matched.extend(glob.glob(pat))

    if not matched:
        raise FileNotFoundError(
            "Aucun fichier 'Runs' trouvé.\n"
            f"- Cherché UNC: {runs_dir}\\{pattern}\n"
            f"- Cherché local: {LOCAL_LITASCO_DIR}\\{pattern}\n"
            "➡️ Dépose le fichier via l'uploader ci-dessous OU copie-le dans "
            "'Litasco balances/Litasco supply' au sein du repo."
        )

    matched.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matched[0]


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

# ================== PLOTTING (Plotly interactif) ==================
def seasonality_figure(series: pd.Series, title: str) -> go.Figure | None:
    """Graphique interactif: courbes par année + bande min/max (2020–2024)."""
    s = series.dropna()
    if s.empty or (s == 0).all():
        return None

    df = pd.DataFrame({"kt": s})
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    # Pivot: lignes=mois, colonnes=années
    ym = (df.groupby(["Year","Month"])["kt"].mean().unstack(level=0).reindex(index=MONTH_TICKS))

    years = [int(y) for y in ym.columns if pd.notna(y)]
    if not years:
        return None

    # Bande 2020–2024
    band_years = [y for y in range(BAND_START, BAND_END + 1) if y in years]
    band_min = band_max = None
    if band_years:
        band_data = ym[band_years]
        band_min = band_data.min(axis=1)
        band_max = band_data.max(axis=1)

    fig = go.Figure()

    # Bande grisée (si dispo)
    if band_years and band_min.notna().any() and band_max.notna().any():
        fig.add_trace(go.Scatter(
            x=MONTH_TICKS, y=band_min.values,
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=MONTH_TICKS, y=band_max.values,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(128,128,128,0.20)",
            name=f"{BAND_START}–{BAND_END} range", hoverinfo="skip"
        ))

    # Courbes par année
    for y in sorted(years):
        line = ym[y]
        color = SPECIAL_COLORS.get(y, None)
        fig.add_trace(go.Scatter(
            x=MONTH_TICKS, y=line.values,
            mode="lines+markers",
            name=str(y),
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

# ================== MAIN TAB RENDERER ==================
def run_litasco_supply_tab():
    st.subheader("Litasco supply — seasonality (interactive)")

    # ---- UI
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        region = st.selectbox("Région", ["NWE", "MED"], index=0, help="MED prêt pour plus tard")
    with c2:
        runs_sheet = st.text_input("Onglet Excel (sheet)", value="NWE" if region == "NWE" else "MED")
    with c3:
        # bouton pour re-générer
        pass

    # Chemins d'entrée (style "Bunker diff/…")
    runs_file_explicit = st.text_input(
        "Fichier Runs (chemin complet facultatif)",
        value=""  # vide => on cherche auto
    )
    runs_dir = st.text_input(
        "Dossier des Runs (UNC Windows)",
        value=DEFAULT_LITASCO_RUNS_DIR
    )
    runs_pattern = st.text_input(
        "Pattern de fichier",
        value="Europe Runs Recap*.xlsx"
    )

    st.caption(f"Fallback local (repo): {LOCAL_LITASCO_DIR}")

    # Uploader (si tu ne peux pas accéder au UNC)
    uploaded = st.file_uploader("…ou dépose un fichier .xlsx ici", type=["xlsx"])
    if uploaded is not None:
        # On sauvegarde dans le dossier local du repo et on l'utilisera comme 'explicit_path'
        save_path = LOCAL_LITASCO_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        runs_file_explicit = str(save_path)
        st.success(f"Fichier uploadé: {save_path.name}")


    # Alerte UNC si non-Windows
    if platform.system() != "Windows" and (runs_dir.startswith("\\") or runs_dir.startswith("//")):
        st.warning(
            "Chemin réseau UNC détecté sur un runtime non-Windows. "
            "Monte le partage réseau ou fournis un fichier local via 'Fichier Runs'."
        )

    st.markdown("---")
    go = st.button("Générer les graphiques")

    if not go:
        st.info("Renseigne les options ci-dessus puis clique **Générer les graphiques**.")
        return

    # ---- Chargement runs
    try:
        runs_path = pick_runs_path(runs_file_explicit, runs_dir, runs_pattern)
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

    # Structures pour affichage
    refinery_product: dict[tuple[str, str], pd.DataFrame] = {}  # (country, refinery) -> df produits
    country_product: dict[str, dict[str, pd.Series]] = {c: {} for c in target_countries}
    totals_product: dict[str, pd.Series | None] = {p: None for p in all_products}

    # ---- Calculs (refinery -> country -> region)
    for country in target_countries:
        runs_c = series_for_country(runs_df, country)
        if not nonempty_series(runs_c):
            continue

        refs = REFINERIES[country]
        total_cap = sum(r["Capacity"] for r in refs)
        if total_cap == 0:
            continue

        util = runs_c / total_cap  # utilisation (kbd/kbd)

        # init pays×produit
        for p in all_products:
            country_product[country][p] = None

        for r in refs:
            Ri = r["Capacity"] * util  # kbd
            prod_cols = {}
            for prod, y in r["Yields"].items():
                series_kt = Ri * y * KBD_TO_KT  # kt
                series_kt.name = prod
                prod_cols[prod] = series_kt

                # accumulate pays×produit
                country_product[country][prod] = (
                    series_kt.copy() if country_product[country][prod] is None
                    else country_product[country][prod].add(series_kt, fill_value=0.0)
                )

                # accumulate région×produit
                totals_product[prod] = (
                    series_kt.copy() if totals_product[prod] is None
                    else totals_product[prod].add(series_kt, fill_value=0.0)
                )

            # compléter colonnes manquantes par 0
            for p in all_products:
                if p not in prod_cols:
                    s0 = pd.Series(0.0, index=Ri.index)
                    s0.name = p
                    prod_cols[p] = s0

            df_ref = pd.DataFrame(prod_cols, index=Ri.index)[all_products]
            refinery_product[(country, r["Refinery"])] = df_ref

    st.success("Graphiques calculés. Affichage…")

    # ---- AFFICHAGE COMME DANS LE MAIL ----
    # 1) Par pays — raffineries (grilles 3 par ligne), puis totaux pays
    for country in sorted(target_countries):
        # ne rien afficher si le pays n'a pas de données
        has_any = any(
            (country, ref["Refinery"]) in refinery_product
            for ref in REFINERIES.get(country, [])
        )
        if not has_any:
            continue

        st.header(country)

        # Raffineries
        with st.expander("Refineries", expanded=False):
            # boucle par raffinerie de ce pays
            refs = REFINERIES[country]
            for ref in refs:
                key = (country, ref["Refinery"])
                if key not in refinery_product:
                    continue
                df_ref = refinery_product[key]
                st.subheader(f"{ref['Refinery']}")
                # grille 3 par ligne
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

        # Totaux pays × produit
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

    # 2) Totaux régionaux
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

    # ---- Export CSV résumé (en mémoire) ----
    today = datetime.now().strftime("%Y-%m-%d")
    summary_rows = []
    # pays
    for country, prod_map in country_product.items():
        for product, s in prod_map.items():
            if s is None:
                continue
            summary_rows.append({
                "Region": region,
                "Country": country,
                "Product": product,
                "Avg_kt": round(float(pd.to_numeric(s, errors="coerce").mean()), 2),
                "Total_kt": round(float(pd.to_numeric(s, errors="coerce").sum()), 2),
            })
    # région
    for product, s in totals_product.items():
        if s is None:
            continue
        summary_rows.append({
            "Region": region,
            "Country": f"{region} Total",
            "Product": product,
            "Avg_kt": round(float(pd.to_numeric(s, errors="coerce").mean()), 2),
            "Total_kt": round(float(pd.to_numeric(s, errors="coerce").sum()), 2),
        })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values(["Region","Country","Product"]).reset_index(drop=True)
        st.subheader("Résumé (tableau)")
        st.dataframe(df_summary, use_container_width=True)

        csv_bytes = df_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "Télécharger le CSV résumé",
            data=csv_bytes,
            file_name=f"Summary_{region}_{today}.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucune donnée non-nulle à résumer.")
