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

def _is_valid_region(col: str) -> bool:
    """Retourne True si 'col' est une vraie région (pas Year/Month/Total/None)."""
    if col is None:
        return False
    c = str(col).strip()
    if not c:
        return False
    cl = c.lower()
    if cl in ("year", "month", "none"):
        return False
    if "total" in cl or "grand" in cl:
        return False
    return True


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
    Devine le libellé de RÉGION porté par chaque colonne de données, même si
    l'entête est éclatée sur 2-3 colonnes horizontales.
    Retour: {index_de_colonne: "Nom de région"} (sans les 'Total', 'None', etc.)
    """
    header_rows = min(header_rows, len(arr))
    ncols = arr.shape[1]

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z]", "", str(s).lower())

    def _window_text(c0: int, c1: int) -> str:
        # c0 inclus, c1 exclus
        parts = []
        for r in range(header_rows):
            for c in range(max(0, c0), min(ncols, c1)):
                cell = str(arr.iat[r, c]).strip()
                if cell and cell.lower() != "none":
                    parts.append(cell)
        return _norm(" ".join(parts))

    col_label: dict[int, str] = {}

    # Pour chaque colonne j, on teste plusieurs fenêtres horizontales
    # (jusqu'à 3 colonnes) qui FINISSENT à j, puis qui DÉBUTENT à j.
    for j in range(ncols):
        found = None

        # fenêtres qui FINISSENT à j  -> couvre le cas "nombre sous la dernière cellule du libellé"
        for width in (3, 2, 1):
            txt = _window_text(j - width + 1, j + 1)
            if not txt:
                continue
            for label, keys in _REGION_KEYS.items():
                if any(k in txt for k in keys):
                    found = label
                    break
            if found:
                break

        # sinon, fenêtres qui DÉBUTENT à j  -> cas plus rare (libellé commence sous cette colonne)
        if not found:
            for width in (1, 2, 3):
                txt = _window_text(j, j + width)
                if not txt:
                    continue
                for label, keys in _REGION_KEYS.items():
                    if any(k in txt for k in keys):
                        found = label
                        break
                if found:
                    break

        # garde uniquement les vraies régions (pas Total / None / Year / Month)
        if found and _is_valid_region(found) and "total" not in found.lower():
            col_label[j] = found

    return col_label


def _find_header_positions(arr: pd.DataFrame, header_row_idx: int, scan_rows: int = 8) -> tuple[int|None, int|None]:
    """
    Retourne (year_col, month_col) en scannant 'scan_rows' lignes d'entête
    à partir de header_row_idx. Renvoie None si non trouvé.
    """
    year_col = None
    month_col = None
    hi = min(len(arr), header_row_idx + scan_rows)
    for r in range(header_row_idx, hi):
        for c in range(arr.shape[1]):
            cell = str(arr.iat[r, c]).strip().lower()
            if not cell or cell == "none":
                continue
            if "year" in cell and year_col is None:
                year_col = c
            if "month" in cell and month_col is None:
                month_col = c
    return year_col, month_col

def _build_region_groups(col_label: dict[int, str]) -> list[tuple[str, list[int]]]:
    """
    Regroupe les colonnes adjacentes qui portent le même libellé de région.
    Retourne une liste de (region_label, [indices_de_colonnes]).
    """
    groups: list[tuple[str, list[int]]] = []
    last_lab = None
    last_idx = []
    for j in sorted(col_label.keys()):
        lab = col_label[j]
        if not _is_valid_region(lab) or "total" in lab.lower():
            # on ignore Total / entêtes parasites
            continue
        if lab == last_lab and last_idx and j == last_idx[-1] + 1:
            last_idx.append(j)            # même région, colonne adjacente -> même groupe
        else:
            if last_lab is not None and last_idx:
                groups.append((last_lab, last_idx))
            last_lab = lab
            last_idx = [j]
    if last_lab is not None and last_idx:
        groups.append((last_lab, last_idx))
    return groups


def _coerce_group_number(row: list[str], idxs: list[int]) -> float:
    """
    Concatène les cellules d'un même groupe, enlève espaces/virgules, puis float().
    Ex: ['3','943',''] -> '3943' -> 3943.0
    """
    raw = "".join(str(row[j]) for j in idxs if j < len(row))
    raw = raw.replace(",", "").replace(" ", "").strip()
    if raw == "" or raw.lower() in {"nan", "none", "—", "-"}:
        return 0.0
    try:
        return float(raw)
    except Exception:
        # dernier filet de sécurité : si ça échoue, retombe sur l'ancienne logique (somme)
        return sum(_coerce_numeric(row[j]) for j in idxs if j < len(row))



def _table_to_monthly(df: pd.DataFrame, fallback_year: str | None = None) -> pd.DataFrame:
    """
    Parse the page-3 'HSFO imports by Load Region' (or LSFO) table into robust
    monthly rows: Year, Month, <regions...>, Total.

    Key choices to avoid the January spike:
    - We only ingest a row when BOTH a Year and a Month are detected on that SAME row.
      (No 'current month' state that could spill across lines.)
    - Region headers often span 1–3 adjacent columns; we detect those blocks and
      read each block as a single number by concatenating numeric fragments.
    - Any row containing 'total' / 'grand total' is skipped.

    Output is sorted by (Year, Month) with a rebuilt 'Total' = sum(region cols).
    """
    # -------- guard rails --------
    if df is None or df.empty:
        return pd.DataFrame()

    # -------- normalise raw table --------
    arr = df.copy().where(pd.notna(df), "").astype(str)
    arr = arr.applymap(lambda s: s.strip())

    # -------- find header row containing 'Year' --------
    header_row_idx = None
    for i in range(min(len(arr), 30)):
        if "year" in [c.lower() for c in arr.iloc[i].tolist()]:
            header_row_idx = i
            break
    if header_row_idx is None:
        header_row_idx = 0

    # -------- scan a small header block to recover labels and key columns --------
    scan_rows = min(8, len(arr) - header_row_idx)
    scan_blk  = arr.iloc[header_row_idx : header_row_idx + scan_rows].copy()

    # map each physical column -> canonical region label (from fuzzy matches)
    col_label = _scan_header_labels(scan_blk, header_rows=len(scan_blk))

    # get the *real* Year and Month column indexes within the header block
    year_col, month_col = _find_header_positions(arr, header_row_idx, scan_rows=scan_rows)

    # -------- build adjacent column groups per region (to read split numbers) ----
    def build_region_groups(col_label: dict[int, str]) -> list[tuple[str, list[int]]]:
        groups: list[tuple[str, list[int]]] = []
        last_lab, last_idxs = None, []
        for j in sorted(col_label.keys()):
            lab = col_label[j]
            if not _is_valid_region(lab) or "total" in lab.lower():
                continue
            if lab == last_lab and last_idxs and j == last_idxs[-1] + 1:
                last_idxs.append(j)
            else:
                if last_lab and last_idxs:
                    groups.append((last_lab, last_idxs))
                last_lab, last_idxs = lab, [j]
        if last_lab and last_idxs:
            groups.append((last_lab, last_idxs))
        return groups

    region_groups = build_region_groups(col_label)

    # -------- helpers to read one logical number from a group of columns ---------
    def coerce_group_number(row: list[str], idxs: list[int]) -> float:
        # join fragments like '3' '943' -> '3943'
        raw = "".join(str(row[j]) for j in idxs if j < len(row))
        raw = raw.replace(",", "").replace(" ", "").strip()
        if raw == "" or raw.lower() in {"nan", "none", "—", "-"}:
            return 0.0
        try:
            return float(raw)
        except Exception:
            # safety fallback: sum of loose parses
            return sum(_coerce_numeric(row[j]) for j in idxs if j < len(row))

    # regex for year in its cell
    year_rx = re.compile(r"(20\d{2})")

    # -------- parse: ONLY rows that have BOTH Year and Month on that row ---------
    records: list[dict] = []
    start_i = header_row_idx + scan_rows
    for i in range(start_i, len(arr)):
        row = arr.iloc[i].tolist()
        rowtxt = " ".join(x.lower() for x in row)

        # drop totals / subtotals
        if "total" in rowtxt:
            continue

        # detect year strictly from the Year column (if available)
        yr = None
        if year_col is not None and year_col < len(row):
            m = year_rx.search(str(row[year_col]))
            if m:
                yr = m.group(1)

        # detect month strictly from the Month column (if available)
        mo = None
        if month_col is not None and month_col < len(row):
            mo = _detect_month_in_row([row[month_col]])

        # if either missing, skip the row (no state carry-over!)
        key_year = yr or fallback_year
        if not key_year or not mo:
            continue

        rec = {"Year": key_year, "Month": mo}
        for lab, idxs in region_groups:
            v = coerce_group_number(row, idxs)
            if v != 0.0:
                rec[lab] = rec.get(lab, 0.0) + v
        records.append(rec)

    if not records:
        return pd.DataFrame()

    # -------- build tidy frame, rebuild Total, sort calendar order --------------
    out = pd.DataFrame(records)
    out.columns = [str(c).strip().replace("\n", " ").replace("  ", " ") for c in out.columns]
    region_cols = [c for c in out.columns if _is_valid_region(c)]
    out["Total"] = out[region_cols].sum(axis=1)

    out["Month"] = pd.Categorical(out["Month"], categories=MONTHS_ORDER, ordered=True)
    out = out.sort_values(["Year", "Month"], kind="stable").reset_index(drop=True)

    # If duplicate (Year, Month) rows still happen (rare PDF quirks), collapse them cleanly
    out = (
        out.groupby(["Year", "Month"], as_index=False)[region_cols + ["Total"]]
           .sum()
           .sort_values(["Year", "Month"])
           .reset_index(drop=True)
    )

    return out[["Year", "Month"] + region_cols + ["Total"]]




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

    # --- Année de secours depuis le nom de fichier (si on ne la trouve pas dans le tableau)
    fallback_year = None
    try:
        fname = Path(chosen_path).name if chosen_path else ""
        dt = _extract_date_from_name(fname)
        if dt:
            fallback_year = str(dt.year)  # ex: "2024"
    except Exception:
        pass
    

    # ---- Extraction des tableaux page 3
    try:
        hsfo_raw, lsfo_raw = _extract_tables_from_pdf(pdf_bytes)
    except Exception as e:
        st.error("Extraction PDF échouée. Tu peux fournir des CSV de secours dans la section 'Fallback CSV' plus bas.")
        with st.expander("Traceback"):
            st.exception(e)
        return

    # Nettoyage / mise en forme (mensuel par régions)
    hsfo_monthly = _table_to_monthly(hsfo_raw, fallback_year=fallback_year)
    lsfo_monthly = _table_to_monthly(lsfo_raw, fallback_year=fallback_year)

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

