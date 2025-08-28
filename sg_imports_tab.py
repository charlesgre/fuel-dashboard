# sg_imports_tab.py
# Onglet: Singapore imports — affichage d'images des pages du PDF (2,3,4,6,7)

from __future__ import annotations
import io
import os
from pathlib import Path
from datetime import datetime
import re

import streamlit as st
import pdfplumber


# --------- utilitaires pour retrouver le dernier PDF dans un dossier ---------

_DATE_PATTERNS = [
    ("%d%m%Y", re.compile(r"(?<!\d)(\d{8})(?!\d)")),  # ddmmyyyy
    ("%Y%m%d", re.compile(r"(?<!\d)(\d{8})(?!\d)")),  # yyyymmdd
]

def _extract_date_from_name(name: str) -> datetime | None:
    for fmt, rx in _DATE_PATTERNS:
        m = rx.search(name)
        if not m:
            continue
        s = m.group(1)
        try:
            dt = datetime.strptime(s, fmt)
            if 2015 <= dt.year <= 2100:
                return dt
        except Exception:
            pass
    return None

def _find_latest_pdf_path(base_dir: str | Path) -> Path | None:
    p = Path(base_dir)
    if not p.exists() or not p.is_dir():
        return None
    candidates = list(p.glob("*.pdf")) + list(p.glob("*.PDF"))
    if not candidates:
        return None

    with_dates, without_dates = [], []
    for f in candidates:
        dt = _extract_date_from_name(f.name)
        (with_dates if dt else without_dates).append((dt, f) if dt else f)

    if with_dates:
        with_dates.sort(key=lambda x: x[0], reverse=True)
        return with_dates[0][1]

    without_dates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return without_dates[0] if without_dates else None


# --------------------------- rendu Streamlit ----------------------------------

def _render_pdf_pages(pdf_bytes: bytes, pages_to_show: list[int], dpi: int = 220) -> None:
    """Affiche les pages demandées (1-indexées) du PDF comme images."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total = len(pdf.pages)
        for pno in pages_to_show:
            st.markdown(f"### Page {pno}")
            if pno < 1 or pno > total:
                st.warning(f"Page {pno} inexistante dans ce PDF (total = {total} pages).")
                continue
            page = pdf.pages[pno - 1]
            # pdfplumber -> PIL.Image via to_image()
            try:
                img = page.to_image(resolution=dpi).original  # PIL.Image
                st.image(img, caption=f"Capture de la page {pno}", use_container_width=True)
            except Exception as e:
                st.error(f"Impossible d'afficher la page {pno} : {e}")


def render_sg_imports_tab(default_dir: str | None = None):
    st.header("🇸🇬 Singapore imports — vues PDF (pages 2, 3, 4, 6, 7)")

    # Dossier par défaut (même logique que précédemment)
    repo_root = Path(os.getenv("FUEL_DASH_DATA_ROOT", Path(__file__).resolve().parent))
    candidates = [
        repo_root / "Singapore hub tracking",
        Path.home() / "Desktop" / "Fuel dashboard" / "Singapore hub tracking",
    ]
    suggested = next((p for p in candidates if p.exists()), candidates[0])
    base_dir = Path(default_dir) if default_dir else suggested

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("PDF source (Power BI export)", type=["pdf"])
    with col2:
        st.caption("Ou bien, laisser l'appli choisir le dernier PDF d’un dossier :")
        base_dir_str = st.text_input("Chemin du dossier", value=str(base_dir))
        base_dir = Path(base_dir_str)

    # Choix de la source PDF (upload prioritaire)
    pdf_bytes = None
    chosen_path = None
    if uploaded is not None:
        pdf_bytes = uploaded.read()
        chosen_path = f"(upload) {uploaded.name}"
    else:
        latest = _find_latest_pdf_path(base_dir)
        if latest is not None:
            try:
                pdf_bytes = latest.read_bytes()
                chosen_path = str(latest)
            except Exception as e:
                st.warning(f"Impossible de lire: {latest}\n{e}")

    if pdf_bytes is None:
        st.info("Aucun PDF fourni. Uploade un fichier ou indique un dossier contenant le PDF.")
        return

    st.caption(f"PDF utilisé : **{chosen_path}**")

    # Pages à afficher (1-indexées) : 2, 3, 4, 6, 7
    pages_to_show = [2, 3, 4, 6, 7]
    _render_pdf_pages(pdf_bytes, pages_to_show, dpi=220)
