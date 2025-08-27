# arbs_tab.py
# Onglet Streamlit pour afficher les tableaux "Arbs" depuis un fichier Excel
# - Affiche UNIQUEMENT les lignes/colonnes non masquées
# - Respecte les cellules fusionnées (rowspan/colspan)
# - Récupère les couleurs, bords et gras principaux pour un rendu proche d'Excel
# - Propose un mode alternatif "éditable" (grid) + export XLSX

from __future__ import annotations
import io
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
import streamlit as st
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries

# --------- CONFIG PAR DÉFAUT ---------
DEFAULT_XLSX_PATH = "Arbs/Arbsheet updated.xlsx"   # adapte si besoin
DEFAULT_SHEET = "MAIN"                             # feuille source par défaut

# -------------------------------------

@dataclass(frozen=True)
class Merge:
    min_col: int
    min_row: int
    max_col: int
    max_row: int

def _hex_color(openpyxl_color) -> str | None:
    """
    Convertit openpyxl Color en #RRGGBB (ignore alpha si 'FF' ou None).
    """
    if not openpyxl_color:
        return None
    try:
        rgb = openpyxl_color.rgb  # ex: 'FFB6D7A8'
        if rgb and len(rgb) == 8:
            return f"#{rgb[2:]}"  # drop alpha
        if rgb and len(rgb) == 6:
            return f"#{rgb}"
    except Exception:
        pass
    return None

def _border_css(b) -> str:
    """
    Renvoie une CSS border résumée à partir d'un Border openpyxl.
    """
    if not b:
        return ""
    sides = []
    for side_name in ("left", "right", "top", "bottom"):
        side = getattr(b, side_name)
        if side and side.style:
            # map styles
            style = "solid"
            width = "1px"
            if side.style in ("medium", "thick"):
                width = "2px"
            elif side.style in ("hair", "dotted", "dashDot"):
                style = "dotted"
            color = _hex_color(side.color) or "#444444"
            sides.append(f"border-{side_name}:{width} {style} {color};")
    return "".join(sides)

def _align_css(al):
    if not al:
        return ""
    parts = []
    if al.horizontal:
        parts.append(f"text-align:{al.horizontal};")
    if al.vertical:
        # vertical -> translate to CSS baseline/middle/bottom
        vmap = {"center": "middle"}.get(al.vertical, al.vertical)
        parts.append(f"vertical-align:{vmap};")
    if al.wrap_text:
        parts.append("white-space:normal;")
    else:
        parts.append("white-space:nowrap;")
    return "".join(parts)

def _font_css(f):
    if not f:
        return ""
    parts = []
    if f.bold:
        parts.append("font-weight:700;")
    if f.italic:
        parts.append("font-style:italic;")
    if f.size:
        parts.append(f"font-size:{f.size}px;")
    color = _hex_color(f.color)
    if color:
        parts.append(f"color:{color};")
    return "".join(parts)

def _fill_css(fill):
    try:
        # only pattern fills with fgColor
        if fill and fill.fgColor:
            color = _hex_color(fill.fgColor)
            if color:
                return f"background-color:{color};"
    except Exception:
        pass
    return ""

def _is_col_hidden(ws, idx: int) -> bool:
    cd = ws.column_dimensions.get(get_column_letter(idx))
    return bool(cd and cd.hidden)

def _is_row_hidden(ws, idx: int) -> bool:
    rd = ws.row_dimensions.get(idx)
    return bool(rd and rd.hidden)

def _visible_bounds(ws):
    """
    Retourne les bornes (min_row, max_row, min_col, max_col) utiles,
    en excluant intégralement les lignes/colonnes masquées aux extrémités.
    """
    max_r = ws.max_row
    max_c = ws.max_column
    # borne min row
    min_r = 1
    while min_r <= max_r and _is_row_hidden(ws, min_r):
        min_r += 1
    # borne max row
    while max_r >= 1 and _is_row_hidden(ws, max_r):
        max_r -= 1
    # borne min col
    min_c = 1
    while min_c <= max_c and _is_col_hidden(ws, min_c):
        min_c += 1
    # borne max col
    while max_c >= 1 and _is_col_hidden(ws, max_c):
        max_c -= 1
    return max(min_r, 1), max_r, max(min_c, 1), max_c

def _collect_merges(ws) -> Dict[Tuple[int, int], Merge]:
    """
    Map (row, col) TOP-LEFT -> Merge(...)
    + set pour tous les (r,c) couverts (afin de sauter les non top-left).
    """
    merges: Dict[Tuple[int,int], Merge] = {}
    for rng in ws.merged_cells.ranges:
        min_c, min_r, max_c, max_r = range_boundaries(str(rng))
        # si la zone contient une ligne/colonne masquée, on ignore la fusion
        hidden = any(_is_row_hidden(ws, r) for r in range(min_r, max_r+1)) or \
                 any(_is_col_hidden(ws, c) for c in range(min_c, max_c+1))
        if hidden:
            continue
        merges[(min_r, min_c)] = Merge(min_c, min_r, max_c, max_r)
    return merges

def _extract_dataframe(ws) -> pd.DataFrame:
    """
    DataFrame "à plat" des cellules visibles (permet mode éditable).
    """
    min_r, max_r, min_c, max_c = _visible_bounds(ws)
    data = []
    for r in range(min_r, max_r+1):
        if _is_row_hidden(ws, r):
            continue
        row_vals = []
        for c in range(min_c, max_c+1):
            if _is_col_hidden(ws, c):
                continue
            row_vals.append(ws.cell(r, c).value)
        data.append(row_vals)
    cols = [get_column_letter(c) for c in range(min_c, max_c+1) if not _is_col_hidden(ws, c)]
    return pd.DataFrame(data, columns=cols)

def _sheet_to_html(ws) -> str:
    """
    Construit un tableau HTML riche qui respecte:
    - colonnes/lignes visibles seulement
    - fusions (rowspan/colspan)
    - couleurs, bords, alignements, gras
    """
    min_r, max_r, min_c, max_c = _visible_bounds(ws)
    merges = _collect_merges(ws)

    # Table CSS de base
    html = [
        '<div style="overflow:auto; max-height:80vh; border:1px solid #ddd; padding:8px;">',
        '<table style="border-collapse:collapse; font-family:Inter, system-ui, Arial; font-size:13px;">'
    ]

    # Pour ignorer les cellules couvertes par une fusion (non top-left)
    covered = set()
    for (tl_r, tl_c), m in merges.items():
        for r in range(m.min_row, m.max_row+1):
            for c in range(m.min_col, m.max_col+1):
                if not (r == tl_r and c == tl_c):
                    covered.add((r, c))

    for r in range(min_r, max_r+1):
        if _is_row_hidden(ws, r):
            continue
        html.append("<tr>")
        for c in range(min_c, max_c+1):
            if _is_col_hidden(ws, c) or (r, c) in covered:
                continue

            cell = ws.cell(r, c)
            # fusion ?
            rs = cs = 1
            m = merges.get((r, c))
            if m:
                rs = m.max_row - m.min_row + 1
                cs = m.max_col - m.min_col + 1

            # styles
            styles = []
            styles.append(_border_css(cell.border))
            styles.append(_align_css(cell.alignment))
            styles.append(_font_css(cell.font))
            styles.append(_fill_css(cell.fill))
            # padding + trait fin par défaut pour mieux coller à Excel
            styles.append("padding:3px 6px;")
            if not cell.border or not any(getattr(cell.border, s).style for s in ("left","right","top","bottom")):
                styles.append("border:1px solid #e0e0e0;")

            value = "" if cell.value is None else str(cell.value)
            # remplace \n par <br> pour multi-lignes
            value = re.sub(r"\n", "<br>", value)

            td = f'<td rowspan="{rs}" colspan="{cs}" style="{"".join(styles)}">{value}</td>'
            html.append(td)
        html.append("</tr>")
    html.append("</table></div>")
    return "".join(html)

# ===================== STREAMLIT TAB =====================

def render():
    st.header("Arbs")

    # Choix du fichier
    path = st.text_input("Chemin du fichier Excel des arbs :", DEFAULT_XLSX_PATH)
    col1, col2 = st.columns([1,1], gap="small")
    editable = col1.toggle("Mode éditable (grid)", value=False, help="Basculer entre rendu 'fidèle Excel' et un grid éditable.")
    show_export = col2.toggle("Activer export XLSX", value=True)

    try:
        wb = load_workbook(path, data_only=True)
    except Exception as e:
        st.error(f"Impossible d’ouvrir le fichier : {e}")
        st.stop()

    # Ne garder que les feuilles visibles
    visible_sheets = [ws.title for ws in wb.worksheets if ws.sheet_state == "visible"]
    if not visible_sheets:
        st.warning("Aucune feuille visible dans ce classeur.")
        st.stop()

    sheet = st.selectbox("Feuille source :", visible_sheets, index=visible_sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in visible_sheets else 0)
    ws = wb[sheet]

    if editable:
        st.caption("Vue éditable (perte des fusions/formatages mais modifiable).")
        df = _extract_dataframe(ws)
        edited = st.data_editor(df, use_container_width=True, height=650)
        if show_export:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                edited.to_excel(writer, sheet_name="Arbs", index=False)
            st.download_button("Exporter la vue éditée en XLSX", data=out.getvalue(),
                               file_name="Arbs_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Rendu fidèle d’Excel (fusions, couleurs, bords).")
        html = _sheet_to_html(ws)
        st.markdown(html, unsafe_allow_html=True)

        if show_export:
            # export de la zone visible telle quelle (valeurs uniquement)
            df = _extract_dataframe(ws)
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="Arbs_visible", index=False)
            st.download_button("Exporter la zone visible en XLSX", data=out.getvalue(),
                               file_name="Arbs_visible.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# Permet d’exécuter ce fichier seul (debug local)
if __name__ == "__main__":
    import streamlit.web.bootstrap
    def main():
        render()
    streamlit.web.bootstrap.run(main, "", [], flag_options={})
