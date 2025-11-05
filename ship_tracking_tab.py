# ship_tracking_tab.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st


# ============================== UI helpers ==============================

def _inject_css():
    st.markdown(
        """
        <style>
        /* resserre un peu la page et l’entête */
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
        /* jolis “chips” */
        .chip {display:inline-block; padding:6px 10px; border-radius:16px;
               background:#f1f3f5; margin-right:8px; font-size:0.92rem;}
        .chip b {opacity:.75}
        /* carte container */
        .card {background:white; border:1px solid #eee; border-radius:14px;
               padding:14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04);}
        /* dataframe arrondi */
        div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden; border:1px solid #eee;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_df_for_display(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Tente d'uniformiser l’affichage: parse dates, colonnes numériques, etc."""
    df2 = df.copy()

    # Parse quelques colonnes clefs si elles existent
    for col in ["ETA", "Updated date", "Updated", "Date"]:
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], errors="coerce")

    for col in ["Quantity"]:
        if col in df2.columns:
            # garder les nombres si possible
            df2[col] = pd.to_numeric(df2[col], errors="ignore")

    # Config Streamlit pour jolies colonnes
    col_config = {}
    if "ETA" in df2.columns:
        col_config["ETA"] = st.column_config.DateColumn("ETA", format="YYYY-MM-DD")
    if "Updated date" in df2.columns:
        col_config["Updated date"] = st.column_config.DateColumn("Updated date", format="YYYY-MM-DD")

    # Colonnes à largeur un peu réduite si présentes
    for short in ["Key", "IMO", "Unit", "BL"]:
        if short in df2.columns:
            col_config[short] = st.column_config.Column(width="small")

    return df2, col_config


# ============================== Main tab ==============================

def render_ship_tracking_tab(project_root: Path | str = ".") -> None:
    project_root = Path(project_root)
    _inject_css()

    # Titre
    st.markdown("## 🛳️ Ship tracking")

    # Sélecteurs
    sel_col1, sel_col2, _ = st.columns([1, 1, 2])
    with sel_col1:
        region = st.selectbox("Région", ["ARA", "MED", "Singapore"], index=0)
    with sel_col2:
        product = st.selectbox("Produit", ["HSFO", "LSFO", "VGO"], index=0)

    # Chips récapitulatives
    st.markdown(
        f'<div class="chip"><b>Région</b>&nbsp; {region}</div>'
        f'<div class="chip"><b>Produit</b>&nbsp; {product}</div>',
        unsafe_allow_html=True,
    )

    # Résolution fichiers
    base = project_root / "Ship tracking" / region
    file_info = _resolve_files(base, product)

    if file_info is None or not file_info.primary.exists():
        st.info("Aucun fichier trouvé pour cette combinaison. Vérifie l’arborescence.")
        return

    # Lecture de la table (feuille la plus à droite, colonnes A:T, jusqu’à Total après HSSR)
    try:
        df_raw, meta = _read_ship_table(file_info.primary)
    except Exception as exc:
        st.error(f"Lecture du fichier impossible: {file_info.primary}\n\n{exc}")
        return

    df, col_config = _format_df_for_display(df_raw)

    st.markdown("### Données (jusqu’à **Total HSSR**)")

    # Carte info + download
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"Fichier: **{file_info.primary.name}**  ·  Feuille: **{meta['sheet']}**")
        with c2:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export CSV", csv, file_name="ship_tracking_table.csv", use_container_width=True)

    # Table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config if col_config else None,
    )

    # Détails (repliable)
    with st.expander("ℹ️ Détails de l’extraction"):
        st.markdown(
            f"""
            <div class="card">
            <b>Chemin</b>: {str(file_info.primary)}<br>
            <b>Feuille la plus à droite</b>: {meta["sheet"]}<br>
            <b>Lignes lues</b>: {len(df)}<br>
            <b>Index 'HSSR'</b>: {meta.get("hssr_row")} ·
            <b>Index 'Total HSSR'</b>: {meta.get("total_hssr_row")}<br>
            <b>Fichiers additionnels</b>:
              {str(file_info.extra1) if file_info.extra1 else "—"} ·
              {str(file_info.extra2) if file_info.extra2 else "—"}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================== I/O helpers ==============================

@dataclass
class FileInfo:
    primary: Path
    extra1: Optional[Path] = None
    extra2: Optional[Path] = None


def _resolve_files(base: Path, product: str) -> Optional[FileInfo]:
    product = product.upper()
    DEFAULT_NAMES: Dict[str, Dict[str, str]] = {
        "HSFO": {
            "table": "Ship tracking import HSFO.xlsx",
            "extra1": "HSFO importation by country.xlsx",
            "extra2": "Perfect datas HSFO 2024-2025.xlsx",
        },
        "LSFO": {"table": "Ship tracking import LSFO.xlsx"},
        "VGO": {"table": "Ship tracking import VGO.xlsx"},
    }
    names = DEFAULT_NAMES.get(product)
    if names is None:
        return None

    folder = base / product
    table = folder / names["table"]
    extra1 = folder / names.get("extra1", "") if names.get("extra1") else None
    extra2 = folder / names.get("extra2", "") if names.get("extra2") else None

    return FileInfo(
        primary=table,
        extra1=extra1 if (extra1 and extra1.exists()) else None,
        extra2=extra2 if (extra2 and extra2.exists()) else None,
    )


def _read_ship_table(xlsx_path: Path) -> Tuple[pd.DataFrame, dict]:
    xlsx_path = Path(xlsx_path)
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet = xls.sheet_names[-1]  # feuille la plus à droite

    df_raw = pd.read_excel(
        xlsx_path, sheet_name=sheet, usecols="A:T", header=0, engine="openpyxl"
    )

    # Recherche des marqueurs
    search = df_raw.astype(str).fillna("")
    hssr_row = None
    for i, row in search.iterrows():
        if any(cell.strip().upper() == "HSSR" for cell in row.values):
            hssr_row = i
            break

    total_hssr_row = None
    for i, row in search.iterrows():
        if hssr_row is not None and i < hssr_row:
            continue
        if any(cell.strip().upper() == "TOTAL" for cell in row.values):
            total_hssr_row = i
            break

    if total_hssr_row is not None:
        df = df_raw.loc[: total_hssr_row].copy()
    else:
        not_all_na = ~df_raw.isna().all(axis=1)
        last = not_all_na[not_all_na].index.max() if not_all_na.any() else len(df_raw) - 1
        df = df_raw.loc[: last].copy()

    meta = {"sheet": sheet, "hssr_row": hssr_row, "total_hssr_row": total_hssr_row}
    return df, meta
