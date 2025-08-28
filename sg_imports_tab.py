# sg_imports_tab.py
# Onglet interactif: Singapore imports (HSFO/LSFO)
# - Extrait les 2 tableaux de la page "HSFO imports by Load Region" / "LSFO imports by Load Region"
# - Recalcule le "Monthly SG HSFO Import (kt)" en sommant par mois (page 3 -> page 2)
# - Affiche tables interactives + téléchargements CSV
# - Propose des uploads CSV pour recréer les graphiques de la page 4 (Russian/Non-Russian, Top 10)

from __future__ import annotations

import io
from typing import Optional, Tuple

import pdfplumber
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path            # NEW
import re                           # NEW
from datetime import datetime       # NEW (utilisé pour parser la date du nom)

TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

MONTHS_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# ---------- Auto-pick du dernier PDF dans un dossier ----------

_DATE_PATTERNS = [
    # 25082025 -> ddmmyyyy
    ("%d%m%Y", re.compile(r"(?<!\d)(\d{8})(?!\d)")),
    # 20250825 -> yyyymmdd
    ("%Y%m%d", re.compile(r"(?<!\d)(\d{8})(?!\d)")),
]

def _extract_date_from_name(name: str) -> datetime | None:
    """Tente d'extraire une date (data-date) du nom de fichier."""
    for fmt, rx in _DATE_PATTERNS:
        m = rx.search(name)
        if not m:
            continue
        s = m.group(1)
        try:
            # Essaie d'abord ddmmyyyy; si incohérent, essaie yyyymmdd, et inversement
            dt = datetime.strptime(s, fmt)
            # filtre un peu: années plausibles
            if 2015 <= dt.year <= 2100:
                return dt
        except Exception:
            continue
    return None

def _find_latest_pdf_path(base_dir: str | Path) -> Path | None:
    """
    Retourne le chemin du PDF le plus 'récent' dans base_dir.
    - priorité à la 'data-date' trouvée dans le nom de fichier (max)
    - sinon, mtime le plus récent
    """
    p = Path(base_dir)
    if not p.exists() or not p.is_dir():
        return None

    candidates = list(p.glob("*.pdf"))
    if not candidates:
        return None

    # d'abord: essaye de classer par date trouvée dans le nom
    with_dates = []
    without_dates = []
    for f in candidates:
        dt = _extract_date_from_name(f.name)
        if dt:
            with_dates.append((dt, f))
        else:
            without_dates.append(f)

    if with_dates:
        # tri par data-date décroissante
        with_dates.sort(key=lambda x: x[0], reverse=True)
        return with_dates[0][1]

    # fallback: par mtime décroissant
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    # Supprime les colonnes vides / dupliquées, normalise les entêtes
    df = df.copy()
    # Enlève colonnes totalement vides
    df = df.loc[:, ~(df.isna() | (df.astype(str).str.strip()=="")).all(0)]
    # La 1ère ligne contient souvent l'entête; on la promeut si "Year" dedans
    header_row_idx = None
    for i in range(min(5, len(df))):
        row = df.iloc[i].astype(str).str.strip().tolist()
        if any(x.lower()=="year" for x in row):
            header_row_idx = i
            break
    if header_row_idx is not None:
        df.columns = df.iloc[header_row_idx].astype(str).str.strip().tolist()
        df = df.iloc[header_row_idx+1:].reset_index(drop=True)
    else:
        # sinon on forge des noms simples
        df.columns = [f"col_{i}" for i in range(1, len(df.columns)+1)]
    # Trim
    df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)
    return df

def _detect_month_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Conserve lignes dont 1ère colonne ∈ {year, month, total}
    first_col = df.columns[0]
    keep = []
    for i, v in enumerate(df[first_col].astype(str).str.strip()):
        vl = v.lower()
        if vl in {"2024","2025","total"} or v in MONTHS_ORDER:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)

def _coerce_numeric(x):
    try:
        # retire virgules, espaces, lettres éventuelles
        s = str(x).replace(",", "").strip()
        if s == "" or s.lower() in {"nan","none","—","-"}:
            return 0.0
        return float(s)
    except Exception:
        return 0.0

def _table_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le tableau 'HSFO/LSFO imports by Load Region' en dataframe:
    index: Month (January..December), colonnes: regions, plus 'Year' & 'Total'
    """
    df = _clean_table(df)
    df = _detect_month_rows(df)
    if df.empty:
        return df

    # identifie la première colonne (Year/Month)
    first_col = df.columns[0]
    # Si la première ligne est '2024' ou '2025', on garde des blocs Année + mois suivants
    records = []
    current_year: Optional[str] = None

    for _, row in df.iterrows():
        label = str(row[first_col]).strip()
        if label in {"2024","2025"}:
            current_year = label
            continue
        if label.lower() == "total":
            # total global: on peut l'ignorer (on recalcule nous-mêmes)
            continue
        if label not in MONTHS_ORDER:
            # ligne non reconnue -> skip
            continue

        rec = {"Year": current_year, "Month": label}
        # le reste des colonnes = régions
        for col in df.columns[1:]:
            val = _coerce_numeric(row[col])
            rec[col] = val
        records.append(rec)

    out = pd.DataFrame(records)
    # Nettoie les noms de colonnes (régions)
    out.columns = [c.strip().replace("\n"," ").replace("  "," ") for c in out.columns]
    # Ajoute Total calculé
    region_cols = [c for c in out.columns if c not in ("Year","Month")]
    if region_cols:
        out["Total"] = out[region_cols].sum(axis=1)
    # Tri par année puis mois
    out["Month"] = pd.Categorical(out["Month"], categories=MONTHS_ORDER, ordered=True)
    out = out.sort_values(["Year","Month"]).reset_index(drop=True)
    return out

def _extract_tables_from_pdf(pdf_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne (hsfo_df_raw, lsfo_df_raw) tels qu'extraits de la page 3.
    Lève ValueError si rien trouvé.
    """
    hsfo_raw = None
    lsfo_raw = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").lower()
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                }
            )
            if not tables:
                continue

            if "hsfo imports by load region" in text:
                # prends la table la plus large
                t = max(tables, key=lambda x: (len(x), len(x[0]) if x else 0))
                hsfo_raw = pd.DataFrame(t)
            if "lsfo imports by load region" in text:
                t = max(tables, key=lambda x: (len(x), len(x[0]) if x else 0))
                lsfo_raw = pd.DataFrame(t)

    if hsfo_raw is None or lsfo_raw is None:
        raise ValueError("Impossible d’extraire les tableaux HSFO/LSFO de la page 3 du PDF.")
    return hsfo_raw, lsfo_raw

def _monthly_total_fig(df_monthly: pd.DataFrame, fuel: str) -> go.Figure:
    """
    Construit le graphe 'Monthly SG {fuel} Import (kt)' en barres empilées par régions,
    + ligne de total.
    """
    title = f"Monthly SG {fuel} Import (kt)"
    # Construit index X: "{Year}-{Month}"
    x = df_monthly["Year"].astype(str) + " " + df_monthly["Month"].astype(str)

    region_cols = [c for c in df_monthly.columns if c not in ("Year","Month","Total")]
    fig = go.Figure()

    for i, col in enumerate(region_cols):
        fig.add_trace(go.Bar(
            x=x,
            y=df_monthly[col],
            name=col,
            marker_color=TAB10[i % len(TAB10)],
            hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.0f}} kt<extra></extra>",
        ))

    # Total en ligne
    fig.add_trace(go.Scatter(
        x=x, y=df_monthly["Total"], mode="lines+markers", name="Total",
    ))

    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Month",
        yaxis_title="Kilotonnes (kt)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="v"),
    )
    return fig

def _style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    # format joli pour les tableaux bruts
    nums = {c: "{:,.0f}" for c in df.columns if c not in ("Year","Month")}
    return df.style.format(nums)

def render_sg_imports_tab(default_dir: str | None = None):
    st.header("🇸🇬 Singapore Fuel Oil Imports (interactive)")

    # 1) Chemin par défaut = Bureau\Fuel dashboard\Singapore hub tracking (Windows)
    suggested = Path.home() / "Desktop" / "Fuel dashboard" / "Singapore hub tracking"
    if default_dir:
        base_dir = Path(default_dir)
    else:
        base_dir = suggested

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("PDF source (Power BI export)", type=["pdf"])
    with col2:
        st.caption("Dossier à scanner (auto-pick dernier PDF)")
        base_dir_str = st.text_input("Folder path", value=str(base_dir))
        base_dir = Path(base_dir_str)

    # 2) Choix de la source PDF (priorité: upload > auto-pick)
    pdf_bytes = None
    chosen_path = None

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        chosen_path = f"(uploaded) {uploaded.name}"
    else:
        latest = _find_latest_pdf_path(base_dir)
        if latest is not None:
            try:
                pdf_bytes = latest.read_bytes()
                chosen_path = str(latest)
            except Exception as e:
                st.warning(f"Impossible de lire: {latest}\n{e}")

    if pdf_bytes is None:
        st.info("Aucun PDF trouvé. Charge un fichier (à gauche) ou vérifie le dossier saisi.")
        return

    st.caption(f"PDF utilisé: **{chosen_path}**")

    # ---- Extraction des tableaux page 3
    try:
        hsfo_raw, lsfo_raw = _extract_tables_from_pdf(pdf_bytes)
    except Exception as e:
        st.error("Extraction PDF échouée. Tu peux fournir des CSV de secours dans la section 'Fallback CSV' plus bas.")
        with st.expander("Traceback"):
            st.exception(e)
        return

    # Nettoyage / mise en forme (mensuel par régions)
    hsfo_monthly = _table_to_monthly(hsfo_raw)
    lsfo_monthly = _table_to_monthly(lsfo_raw)

    # =============== PAGE 2 cible: Monthly SG HSFO Import
    st.subheader("Monthly SG HSFO Import (kt) — interactive (reconstruit depuis la page 3)")
    fig_hsfo = _monthly_total_fig(hsfo_monthly, fuel="HSFO")
    st.plotly_chart(fig_hsfo, use_container_width=True)

    # (Bonus) Monthly SG LSFO Import — même logique
    with st.expander("Monthly SG LSFO Import (kt) — bonus"):
        fig_lsfo = _monthly_total_fig(lsfo_monthly, fuel="LSFO")
        st.plotly_chart(fig_lsfo, use_container_width=True)

    # =============== PAGE 3: tableaux bruts
    st.markdown("---")
    st.subheader("Page 3 — HSFO imports by Load Region (nettoyé)")
    st.dataframe(_style_table(hsfo_monthly), use_container_width=True)
    st.download_button("Download HSFO monthly (CSV)",
                       hsfo_monthly.to_csv(index=False).encode(),
                       file_name="sg_hsfo_monthly_by_region.csv",
                       mime="text/csv")

    st.subheader("Page 3 — LSFO imports by Load Region (nettoyé)")
    st.dataframe(_style_table(lsfo_monthly), use_container_width=True)
    st.download_button("Download LSFO monthly (CSV)",
                       lsfo_monthly.to_csv(index=False).encode(),
                       file_name="sg_lsfo_monthly_by_region.csv",
                       mime="text/csv")

    # =============== PAGE 4: graphes (via CSV optionnels)
    st.markdown("---")
    st.subheader("Page 4 — Graphes (Russian / Non-Russian, Top exporters)")
    st.caption("Le PDF ne fournit pas ces données sous forme de tableau. "
               "Charge des CSV pour générer ces graphes.")

    with st.expander("Uploader CSV — Russian vs Non-Russian (mensuel)"):
        st.write("Schéma attendu: columns = Year, Month, Fuel, OriginFlag, Volume_kt")
        st.write("Ex: 2025, September, HSFO, Russian, 1710")
        csv_rr = st.file_uploader("CSV Russian/Non-Russian", type=["csv"], key="csv_rr")
        if csv_rr:
            rr = pd.read_csv(csv_rr)
            rr["Month"] = pd.Categorical(rr["Month"], categories=MONTHS_ORDER, ordered=True)
            rr = rr.sort_values(["Year","Month"])
            x = rr["Year"].astype(str) + " " + rr["Month"].astype(str) + " " + rr["Fuel"]
            fig = go.Figure()
            for i, flag in enumerate(rr["OriginFlag"].unique()):
                sel = rr[rr["OriginFlag"]==flag]
                fig.add_trace(go.Bar(
                    x=x, y=sel["Volume_kt"], name=str(flag),
                    marker_color=TAB10[i % len(TAB10)]
                ))
            fig.update_layout(barmode="stack", title="Singapore imports by Origin (kt)")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Uploader CSV — Top exporters (bar chart)"):
        st.write("Schéma attendu: columns = Year, Fuel, Exporter, Volume_kt")
        csv_top = st.file_uploader("CSV Top exporters", type=["csv"], key="csv_top")
        if csv_top:
            top = pd.read_csv(csv_top)
            for (yr, fuel), grp in top.groupby(["Year","Fuel"]):
                show = grp.sort_values("Volume_kt", ascending=False).head(10)
                fig = go.Figure([go.Bar(
                    x=show["Volume_kt"], y=show["Exporter"],
                    orientation="h", marker_color=TAB10[0]
                )])
                fig.update_layout(title=f"Top 10 {fuel} exporters to Singapore — {yr}",
                                  xaxis_title="kt")
                st.plotly_chart(fig, use_container_width=True)

