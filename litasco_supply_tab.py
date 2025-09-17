# -*- coding: utf-8 -*-
import os, glob, platform, re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------- Dossiers ----------
DEFAULT_LITASCO_RUNS_DIR = r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\Litasco balances\Litasco supply"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LITASCO_DIR = REPO_ROOT / "Litasco balances" / "Litasco supply"
LOCAL_LITASCO_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Paramétrage raffineries ----------
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
        "Finland": [{"Refinery": "Porvoo", "Capacity": 96, "Yields": {"VGO": 0.03}}],
        "Ireland": [{"Refinery": "Whitegate", "Capacity": 75, "Yields": {"LSSR": 0.16}}],
        "Lithuania": [{"Refinery": "Mazeikiai", "Capacity": 200, "Yields": {"HSFO": 0.08}}],
        "Poland": [
            {"Refinery": "Gdansk", "Capacity": 210, "Yields": {"HSFO": 0.02}},
            {"Refinery": "Plock",  "Capacity": 360, "Yields": {"HSFO": 0.02}},
        ],
    },
    "MED": {}
}

# ---------- Constantes ----------
KBD_TO_KT = 6.35
BAND_START, BAND_END = 2020, 2024
SPECIAL_COLORS = {2026: "red", 2025: "black", 2024: "green"}
MONTH_TICKS = list(range(1, 13))
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ================== IO helpers ==================
_DATE_PAT = re.compile(r"(\d{2}[.\-]\d{2}[.\-]\d{4}|\d{8}|\d{4}-\d{2}-\d{2})")

def _parse_date_from_name(p: Path):
    m = _DATE_PAT.search(p.name)
    if not m:
        return None
    s = m.group(1)
    for fmt in ("%m.%d.%Y", "%d.%m.%Y", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def _list_matches(dir_path: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(dir_path.glob(pat))
    return list({p.resolve() for p in out if p.is_file()})

def pick_latest_runs_path(runs_dir: str) -> str:
    """
    Choisit automatiquement le dernier Excel :
      1) dans runs_dir (UNC),
      2) sinon dans LOCAL_LITASCO_DIR (copie locale).
    Try patterns: 'Europe Runs Recap*.xlsx' puis '*.xlsx'.
    Tri: date dans le nom > date de modification.
    """
    patterns = ["Europe Runs Recap*.xlsx", "*.xlsx"]

    candidates: list[Path] = []
    # 1) UNC (si fourni)
    if runs_dir:
        dir_unc = Path(runs_dir)
        candidates += _list_matches(dir_unc, patterns)

    # 2) Fallback local du repo
    candidates += _list_matches(LOCAL_LITASCO_DIR, patterns)

    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier Excel trouvé dans:\n"
            f"- {runs_dir}\n- {LOCAL_LITASCO_DIR}\n"
            f"(patterns testés: {', '.join(patterns)})"
        )

    # Trier par date dans le nom (si dispo), sinon mtime
    scored = []
    for p in candidates:
        d = _parse_date_from_name(p)
        scored.append((p, d if d is not None else datetime.fromtimestamp(p.stat().st_mtime)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return str(scored[0][0])

def load_runs(runs_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(runs_path, sheet_name=sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")

def series_for_country(runs_df: pd.DataFrame, country: str):
    if country in runs_df.columns:
        return runs_df[country]
    if country == "United Kingdom" and "UK" in runs_df.columns:
        return runs_df["UK"]
    return None

def nonempty_series(s: pd.Series | None) -> bool:
    if s is None: return False
    vals = pd.to_numeric(s, errors="coerce")
    return not (vals.isna().all() or np.nan_to_num(vals.values).sum() == 0.0)

# ================== Plotly ==================
def seasonality_figure(series: pd.Series, title: str) -> go.Figure | None:
    s = series.dropna()
    if s.empty or (s == 0).all():
        return None

    df = pd.DataFrame({"kt": s})
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    ym = (df.groupby(["Year","Month"])["kt"].mean()
            .unstack(level=0).reindex(index=MONTH_TICKS))

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

    # Bande min/max
    if band_years and band_min.notna().any() and band_max.notna().any():
        fig.add_trace(go.Scatter(
            x=MONTH_TICKS, y=band_min.values, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=MONTH_TICKS, y=band_max.values, mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(128,128,128,0.20)",
            name=f"{BAND_START}–{BAND_END} range", hoverinfo="skip"
        ))

    # Courbes par année
    for y in sorted(years):
        line = ym[y]
        fig.add_trace(go.Scatter(
            x=MONTH_TICKS, y=line.values, mode="lines+markers",
            name=str(y),
            line=dict(width=2 if y in SPECIAL_COLORS else 1.3,
                      color=SPECIAL_COLORS.get(y, None)),
            opacity=1.0 if y in SPECIAL_COLORS else 0.65,
            marker=dict(size=4)
        ))

    # Légende SOUS le graphe + marges adaptées
    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=MONTH_TICKS, ticktext=MONTH_LABELS),
        yaxis_title="kt",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.22,   # sous la figure
            xanchor="left", x=0.0,
            font=dict(size=10)
        ),
        margin=dict(l=10, r=10, t=40, b=90),  # + de marge en bas pour la légende
        height=420
    )
    return fig

def render_fig_grid(figs: list[go.Figure | None], max_cols: int = 3, key_prefix: str = "fig"):
    """Affiche les figures en 1..max_cols colonnes, centrées si < max_cols,
    en donnant une clé unique à chaque chart pour éviter StreamlitDuplicateElementId.
    """
    valid = [f for f in figs if f is not None]
    if not valid:
        return
    n = len(valid)

    # 1 figure : centrée
    if n == 1:
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.plotly_chart(
                valid[0],
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"{key_prefix}-0",
            )
        return

    # 2 figures : 2 colonnes, sinon jusqu'à 3 colonnes
    cols_count = 2 if n == 2 else min(max_cols, 3)
    rows = (n + cols_count - 1) // cols_count
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_count)
        for c in range(cols_count):
            if idx >= n:
                break
            with cols[c]:
                st.plotly_chart(
                    valid[idx],
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"{key_prefix}-{idx}",
                )
            idx += 1



# ================== TAB ==================
def run_litasco_supply_tab():
    st.subheader("Litasco supply — seasonality (interactive)")

    c1, c2 = st.columns([1, 1])
    with c1:
        region = st.selectbox("Région", ["NWE", "MED"], index=0)
    with c2:
        runs_sheet = st.text_input("Onglet Excel (sheet)", value=("NWE" if region == "NWE" else "MED"))

    runs_dir = st.text_input("Dossier des Runs (UNC ou vide pour utiliser le dossier local du repo)",
                             value=DEFAULT_LITASCO_RUNS_DIR)

    if platform.system() != "Windows" and (runs_dir.startswith("\\") or runs_dir.startswith("//")):
        st.warning("Chemin UNC détecté sur un runtime non-Windows. Le code basculera sur le dossier local du repo si le partage n’est pas monté.")

    st.caption(f"Dossier local du repo : {LOCAL_LITASCO_DIR}")

    st.markdown("---")
    if not st.button("Générer les graphiques"):
        st.info("Clique sur **Générer les graphiques**.")
        return

    # 1) Fichier: prend automatiquement le plus récent
    try:
        runs_path = pick_latest_runs_path(runs_dir)
        st.success(f"Fichier sélectionné automatiquement : **{os.path.basename(runs_path)}**")
        st.caption(runs_path)
    except Exception as e:
        st.error(f"Impossible de trouver un Excel : {e}")
        return

    # 2) Lecture
    try:
        runs_df = load_runs(runs_path, runs_sheet)
    except Exception as e:
        st.error(f"Lecture Excel : {e}")
        return

    REFINERIES = REFINERIES_BY_REGION.get(region, {})
    if not REFINERIES:
        st.warning(f"Aucune raffinerie configurée pour {region}.")
        return

    target_countries = list(REFINERIES.keys())
    all_products = sorted({p for plist in REFINERIES.values() for r in plist for p in r["Yields"].keys()})

    refinery_product: dict[tuple[str, str], pd.DataFrame] = {}
    country_product: dict[str, dict[str, pd.Series]] = {c: {} for c in target_countries}
    totals_product: dict[str, pd.Series | None] = {p: None for p in all_products}

    # 3) Calculs
    for country in target_countries:
        runs_c = series_for_country(runs_df, country)
        if not nonempty_series(runs_c):
            continue

        refs = REFINERIES[country]
        total_cap = sum(r["Capacity"] for r in refs)
        if total_cap == 0:
            continue

        util = runs_c / total_cap

        for p in all_products:
            country_product[country][p] = None

        for r in refs:
            Ri = r["Capacity"] * util
            prod_cols = {}
            for prod, y in r["Yields"].items():
                s_kt = Ri * y * KBD_TO_KT
                s_kt.name = prod
                prod_cols[prod] = s_kt

                country_product[country][prod] = (
                    s_kt.copy() if country_product[country][prod] is None
                    else country_product[country][prod].add(s_kt, fill_value=0.0)
                )
                totals_product[prod] = (
                    s_kt.copy() if totals_product[prod] is None
                    else totals_product[prod].add(s_kt, fill_value=0.0)
                )

            for p in all_products:
                if p not in prod_cols:
                    prod_cols[p] = pd.Series(0.0, index=Ri.index, name=p)

            df_ref = pd.DataFrame(prod_cols, index=Ri.index)[all_products]
            refinery_product[(country, r["Refinery"])] = df_ref

    # 4) Affichage
    for country in sorted(target_countries):
        if not any((country, ref["Refinery"]) in refinery_product for ref in REFINERIES.get(country, [])):
            continue

        st.header(country)

        with st.expander("Refineries", expanded=False):
            for ref in REFINERIES[country]:
                key = (country, ref["Refinery"])
                if key not in refinery_product:
                    continue
                df_ref = refinery_product[key]
                st.subheader(f"{ref['Refinery']}")
                figs_ref = []
                for product in df_ref.columns:
                    s = df_ref[product]
                    if s.isna().all() or (s == 0).all():
                        continue
                    figs_ref.append(seasonality_figure(s, f"{country} / {ref['Refinery']} / {product}"))
                render_fig_grid(figs_ref, max_cols=3, key_prefix=f"ref-{country}-{ref['Refinery']}")


        st.subheader("Country totals")
        figs_ctry = []
        for product, s in country_product[country].items():
            if s is None or s.isna().all() or (np.nan_to_num(s.values).sum() == 0.0):
                continue
            figs_ctry.append(seasonality_figure(s, f"{country} / {product}"))
        render_fig_grid(figs_ctry, max_cols=3, key_prefix=f"ctry-{country}")


        st.markdown("---")

    # --- Totaux régionaux (une seule fois, hors de la boucle pays)
    st.header(f"{region} — Totaux régionaux")
    figs_reg = []
    for product, s in sorted(totals_product.items(), key=lambda kv: kv[0].lower()):
        if s is None or s.isna().all() or (np.nan_to_num(s.values).sum() == 0.0):
            continue
        figs_reg.append(seasonality_figure(s, f"{region} Total / {product}"))
    render_fig_grid(figs_reg, max_cols=3, key_prefix=f"region-{region}")


    # 5) Résumé
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for country, prod_map in country_product.items():
        for product, s in prod_map.items():
            if s is None: continue
            rows.append({
                "Region": region, "Country": country, "Product": product,
                "Avg_kt": round(float(pd.to_numeric(s, errors="coerce").mean()), 2),
                "Total_kt": round(float(pd.to_numeric(s, errors="coerce").sum()), 2),
            })
    for product, s in totals_product.items():
        if s is None: continue
        rows.append({
            "Region": region, "Country": f"{region} Total", "Product": product,
            "Avg_kt": round(float(pd.to_numeric(s, errors="coerce").mean()), 2),
            "Total_kt": round(float(pd.to_numeric(s, errors="coerce").sum()), 2),
        })

    if rows:
        df_summary = pd.DataFrame(rows).sort_values(["Region","Country","Product"]).reset_index(drop=True)
        st.subheader("Résumé (tableau)")
        st.dataframe(df_summary, use_container_width=True)
        st.download_button(
            "Télécharger le CSV résumé",
            data=df_summary.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"Summary_{region}_{today}.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucune donnée non-nulle à résumer.")
