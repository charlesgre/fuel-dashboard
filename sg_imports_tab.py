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
import os 

TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

MONTHS_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# Mois "fuzzy" (on acceptera Jan/Janua…, Nov/Novemb…, etc.)
MONTH_PREFIX = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}

# Libellés attendus (avec mots-clés pour les repérer dans des entêtes cassées)
_REGION_KEYS = {
    "Caribbean, Central America": ["caribbean", "centralamerica"],
    "Mediterranean, North Africa": ["mediterranean", "northafrica"],
    "Middle East": ["middleeast"],
    "North America": ["northamerica"],
    "North Asia": ["northasia"],
    "Other Africa": ["otherafrica"],
    "Pacific": ["pacific"],
    "Russia, FSU": ["russia", "fsu"],
    "South America": ["southamerica"],
    "South Asia": ["southasia"],
    "South East Asia": ["southeastasia", "south east asia"],
    "Total": ["total"],
}

def _norm(s: str) -> str:
    """Minuscule + seulement lettres (utile pour matcher des entêtes cassées)."""
    return re.sub(r"[^a-z]", "", str(s).lower())

def _detect_month_in_row(cells: list[str]) -> str | None:
    """Retourne le mois canonique si une cellule commence par Jan/Feb/..."""
    for c in cells[:8]:
        n = _norm(c)
        for pref, full in MONTH_PREFIX.items():
            if n.startswith(pref):
                return full
    return None

def _find_year_in_row(cells: list[str]) -> str | None:
    """Repère 20xx dans les 6 premières cellules."""
    for c in cells[:6]:
        m = re.search(r"(20\d{2})", str(c))
        if m:
            return m.group(1)
    return None

def _scan_header_labels(arr: pd.DataFrame, header_rows: int = 6) -> dict[int, str]:
    """
    Regarde les 'header_rows' premières lignes et devine, pour chaque colonne,
    quel libellé de région elle porte (même si l'entête est sur 2 lignes ou bruitée).
    """
    header_rows = min(header_rows, len(arr))
    col_label: dict[int, str] = {}
    for j in range(arr.shape[1]):
        # concatène les cellules d'entête verticalement pour la colonne j
        txt = " ".join(
            str(arr.iat[i, j]) for i in range(header_rows)
            if str(arr.iat[i, j]).strip() not in ("", "None")
        )
        n = _norm(txt)
        if not n:
            continue
        for label, keys in _REGION_KEYS.items():
            if any(k in n for k in keys):
                col_label[j] = label
                break
    return col_label



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
    p = Path(base_dir)
    if not p.exists() or not p.is_dir():
        return None

    # accepte .pdf et .PDF
    candidates = list(p.glob("*.pdf")) + list(p.glob("*.PDF"))
    if not candidates:
        return None

    with_dates, without_dates = [], []
    for f in candidates:
        dt = _extract_date_from_name(f.name)  # ex. 25082025 dans le nom
        if dt:
            with_dates.append((dt, f))
        else:
            without_dates.append(f)

    if with_dates:
        # tri par date de données (dans le nom) décroissante
        with_dates.sort(key=lambda x: x[0], reverse=True)
        return with_dates[0][1]

    # sinon, on prend le fichier au mtime le plus récent
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]

def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    # Supprime les colonnes vides / dupliquées, normalise les entêtes
    df = df.copy()

    # 1) Enlève les colonnes totalement vides (NaN ou chaînes vides après trim)
    stripped0 = df.astype(str).applymap(lambda s: s.strip() if pd.notna(s) else s)
    non_empty_cols = ~(df.isna() | (stripped0 == "")).all(axis=0)
    df = df.loc[:, non_empty_cols]

    # 2) La 1ère ligne contient souvent l'entête; on la promeut si "Year" dedans
    header_row_idx = None
    for i in range(min(5, len(df))):
        row = df.iloc[i].astype(str).str.strip().tolist()
        if any(x.lower() == "year" for x in row):
            header_row_idx = i
            break
    if header_row_idx is not None:
        df.columns = df.iloc[header_row_idx].astype(str).str.strip().tolist()
        df = df.iloc[header_row_idx + 1 :].reset_index(drop=True)
    else:
        # sinon on forge des noms simples
        df.columns = [f"col_{i}" for i in range(1, len(df.columns) + 1)]

    # 2bis) ⚠️ Supprime les colonnes portant le même nom (on garde la 1re)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    # 3) Trim final des cellules
    df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)
    return df



def _detect_month_rows(df: pd.DataFrame) -> pd.DataFrame:
    # On lit la 1re colonne par POSITION pour éviter le problème de doublons
    s = df.iloc[:, 0].astype(str).str.strip()
    keep = []
    for i, v in enumerate(s):
        vl = v.lower()
        if vl in {"2024", "2025", "total"} or v in MONTHS_ORDER:
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
    Convertit la table brute (page 3) en lignes mensuelles:
    colonnes = Year, Month, <régions...>, Total.
    Gère les blocs multi-lignes par mois et les en-têtes partiellement vides.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Tout en chaînes nettoyées
    arr = df.copy()
    arr = arr.where(pd.notna(arr), "").astype(str)
    arr = arr.applymap(lambda s: s.strip())

    # 1) Trouver la ligne d'entête (celle qui contient 'Year')
    header_row_idx = None
    for i in range(min(len(arr), 15)):
        row_lower = [c.lower() for c in arr.iloc[i].tolist()]
        if "year" in row_lower:
            header_row_idx = i
            break
    if header_row_idx is None:
        return pd.DataFrame()

    # 2) Reconstruire les noms de colonnes (propage le dernier nom non vide)
    base_hdr = arr.iloc[header_row_idx].tolist()
    next_hdr = arr.iloc[header_row_idx + 1].tolist() if header_row_idx + 1 < len(arr) else [""] * arr.shape[1]
    cols, last = [], ""
    for j, cell in enumerate(base_hdr):
        name = cell if cell not in ("", "None") else next_hdr[j]
        name = (name or "").strip()
        if name.lower() == "year":
            cols.append("Year")
            last = ""
        else:
            if name:
                last = name
            cols.append(last)

    if all(c == "" for c in cols):
        return pd.DataFrame()

    # 3) Parcours & agrégation par (Year, Month) sur toutes les lignes du bloc
    year_rx = re.compile(r"\b(20\d{2})\b")
    current_year: Optional[str] = None
    current_month: Optional[str] = None
    acc: dict[tuple[str, str], dict[str, float]] = {}

    for i in range(header_row_idx + 1, len(arr)):
        row = arr.iloc[i].tolist()

        # met à jour l'année si on la voit (souvent une ligne seule)
        yr = None
        for c in row[:6]:
            m = year_rx.search(c)
            if m:
                yr = m.group(1)
                break
        if yr:
            current_year = yr  # on n'interrompt pas: la ligne peut contenir aussi des nombres

        # détecte le mois si présent sur cette ligne; sinon, on reste sur le mois courant
        month = None
        for c in row[:6]:
            if c in MONTHS_ORDER:
                month = c
                break
        if month:
            current_month = month

        # si on n'a pas encore Année + Mois, on ignore
        if not current_year or not current_month:
            continue

        key = (current_year, current_month)
        if key not in acc:
            acc[key] = {}

        # agrège les valeurs régionales de la ligne courante
        for j, val in enumerate(row):
            name = cols[j] if j < len(cols) else ""
            if not name or name == "Year":
                continue
            v = _coerce_numeric(val)
            if v == 0:
                continue
            acc[key][name] = acc[key].get(name, 0.0) + v

    # 4) DataFrame final
    records = []
    for (yr, mo), d in acc.items():
        rec = {"Year": yr, "Month": mo}
        rec.update(d)
        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out

    # Nettoyage colonnes + Total
    out.columns = [str(c).strip().replace("\n", " ").replace("  ", " ") for c in out.columns]
    region_cols = [c for c in out.columns if c not in ("Year", "Month", "Total")]
    out["Total"] = out[region_cols].sum(axis=1)

    # Tri Year / Month
    out["Month"] = pd.Categorical(out["Month"], categories=MONTHS_ORDER, ordered=True)
    out = out.sort_values(["Year", "Month"]).reset_index(drop=True)
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
    + ligne de total. Robuste si le parsing PDF renvoie un DF vide ou sans Year/Month.
    """
    title = f"Monthly SG {fuel} Import (kt)"
    fig = go.Figure()

    # Garde-fous : DF vide ou colonnes essentielles manquantes
    if (
        df_monthly is None
        or df_monthly.empty
        or "Year" not in df_monthly.columns
        or "Month" not in df_monthly.columns
    ):
        fig.add_annotation(
            text="No monthly data parsed from PDF",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Kilotonnes (kt)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="v"),
        )
        return fig

    # Copie triée par Year puis Month (ordre calendaire)
    tmp = df_monthly.copy()
    try:
        tmp["__MonthCat"] = pd.Categorical(
            tmp["Month"].astype(str),
            categories=MONTHS_ORDER,
            ordered=True
        )
        tmp = tmp.sort_values(["Year", "__MonthCat"]).drop(columns="__MonthCat")
    except Exception:
        # si Month ne matche pas exactement, on garde l'ordre original
        pass

    # Colonnes régions (tout sauf Year/Month/Total)
    region_cols = [c for c in tmp.columns if c not in ("Year", "Month", "Total")]

    # Forcer numérique et recalculer Total si absent
    for c in region_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
    if "Total" not in tmp.columns:
        tmp["Total"] = tmp[region_cols].sum(axis=1) if region_cols else 0.0

    # Axe X : "YYYY Month"
    x = tmp["Year"].astype(str) + " " + tmp["Month"].astype(str)

    # Barres empilées par région (palette TAB10)
    for i, col in enumerate(region_cols):
        fig.add_trace(go.Bar(
            x=x,
            y=tmp[col],
            name=col,
            marker_color=TAB10[i % len(TAB10)],
            hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.0f}} kt<extra></extra>",
        ))

    # Ligne du total
    fig.add_trace(go.Scatter(
        x=x,
        y=tmp["Total"],
        mode="lines+markers",
        name="Total",
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

    # 1) Default dir = repo_root / "Singapore hub tracking"  (fallback = Desktop)
    repo_root = Path(os.getenv("FUEL_DASH_DATA_ROOT", Path(__file__).resolve().parent))
    candidates = [
        repo_root / "Singapore hub tracking",                                  # <- dossier du repo
        Path.home() / "Desktop" / "Fuel dashboard" / "Singapore hub tracking", # fallback Windows
    ]
    # on prend le 1er qui existe
    suggested = next((p for p in candidates if p.exists()), candidates[0])

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

    if hsfo_monthly.empty:
        st.warning("Aucune ligne mensuelle HSFO n'a été détectée dans la page 3. "
                "Aperçu du tableau brut ci-dessous (debug).")
        st.dataframe(hsfo_raw.head(20), use_container_width=True)

    if lsfo_monthly.empty:
        st.info("Aucune ligne mensuelle LSFO détectée (facultatif).")

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

