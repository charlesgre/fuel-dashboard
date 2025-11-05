# ship_tracking_tab.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Point d’entrée appelé depuis ton app principale
# ---------------------------------------------------------------------------
def render_ship_tracking_tab(project_root: Path | str = ".") -> None:
    """
    Onglet Streamlit: Ship tracking.

    Parameters
    ----------
    project_root : Path | str
        Chemin vers le dossier racine du projet "Fuel dashboard".
        Ex: r"C:/Users/<YOU>/Desktop/Fuel dashboard"
    """
    project_root = Path(project_root)

    st.header("Ship tracking")

    # Sélecteurs
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Région", ["ARA", "MED", "Singapore"], index=0)
    with col2:
        product = st.selectbox("Produit", ["HSFO", "LSFO", "VGO"], index=0)

    # Résolution des fichiers à partir de l'arborescence montrée
    base = project_root / "Ship tracking" / region
    file_info = _resolve_files(base, product)

    if file_info is None or not file_info.primary.exists():
        st.info(
            "Aucun fichier trouvé pour cette combinaison. "
            "Vérifie l'arborescence ou mets à jour les noms dans _resolve_files()."
        )
        return

    # --- Implémentation de la 1ʳᵉ étape demandée ---
    # Lire la 1ʳᵉ table depuis le fichier principal :
    # - feuille la plus à droite
    # - colonnes A:T
    # - lignes jusqu'à 'Total' après la ligne contenant 'HSSR'
    try:
        df_display, meta = _read_ship_table(file_info.primary)
    except Exception as exc:
        st.error(f"Lecture du fichier impossible: {file_info.primary}\n\n{exc}")
        return

    st.subheader("Données (jusqu'à 'Total HSSR')")
    st.caption(f"Fichier: {file_info.primary.name} — Feuille: {meta['sheet']}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    with st.expander("Détails de l'extraction"):
        st.write(
            {
                "Chemin": str(file_info.primary),
                "Feuille la plus à droite": meta["sheet"],
                "Lignes lues": len(df_display),
                "Colonnes (A:T)": list(df_display.columns),
                "Index 'HSSR'": meta.get("hssr_row"),
                "Index 'Total HSSR'": meta.get("total_hssr_row"),
                "Extra 1": str(file_info.extra1) if file_info.extra1 else None,
                "Extra 2": str(file_info.extra2) if file_info.extra2 else None,
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class FileInfo:
    primary: Path
    extra1: Optional[Path] = None
    extra2: Optional[Path] = None


def _resolve_files(base: Path, product: str) -> Optional[FileInfo]:
    """
    Retourne les fichiers à utiliser pour (region, product).

    Cas particulier décrit: ARA / HSFO utilise 3 fichiers (table + 2 autres
    pour graphiques/tableaux). Ici on lit seulement le fichier "table", mais on
    expose déjà les autres chemins pour la suite.
    """
    product = product.upper()

    # Nommage par défaut (adapte si tes noms diffèrent)
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


def _read_ship_table(xlsx_path: Path) -> Tuple[pd.DataFrame, Dict[str, int | str | None]]:
    """
    Lit la feuille la plus à droite d'un Excel et renvoie la table:

    - Colonnes limitées à A:T (usecols="A:T")
    - Recherche la première ligne contenant 'HSSR' (n'importe quelle colonne)
      puis la première ligne *après* ça contenant 'Total'; on coupe jusqu'à
      cette ligne 'Total' (incluse).
    - Si les marqueurs manquent, on supprime juste les lignes vides en bas.
    """
    xlsx_path = Path(xlsx_path)
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet = xls.sheet_names[-1]  # feuille la plus à droite

    # Lecture brute; header=0 suppose que la ligne d'entête est la première non vide
    df_raw = pd.read_excel(
        xlsx_path, sheet_name=sheet, usecols="A:T", header=0, engine="openpyxl"
    )

    # Recherche des marqueurs
    df_search = df_raw.astype(str).fillna("")
    hssr_row = None
    for i, row in df_search.iterrows():
        if any(cell.strip().upper() == "HSSR" for cell in row.values):
            hssr_row = i
            break

    total_hssr_row = None
    for i, row in df_search.iterrows():
        if hssr_row is not None and i < hssr_row:
            continue
        if any(cell.strip().upper() == "TOTAL" for cell in row.values):
            total_hssr_row = i
            break

    # Découpage des lignes
    if total_hssr_row is not None:
        df = df_raw.loc[:total_hssr_row].copy()
    else:
        not_all_na = ~df_raw.isna().all(axis=1)
        last = not_all_na[not_all_na].index.max() if not_all_na.any() else len(df_raw) - 1
        df = df_raw.loc[:last].copy()

    meta = {"sheet": sheet, "hssr_row": hssr_row, "total_hssr_row": total_hssr_row}
    return df, meta
