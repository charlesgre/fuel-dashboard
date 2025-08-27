# arbs_tab.py
from __future__ import annotations
import io, re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
import streamlit as st
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries

DEFAULT_XLSX_PATH = "Arbs/Arbsheet updated.xlsx"
DEFAULT_SHEET = "MAIN"

@dataclass(frozen=True)
class Merge:
    min_col: int
    min_row: int
    max_col: int
    max_row: int

# ---------- helpers couleurs ----------
def _hex_color(c) -> str | None:
    if not c:
        return None
    try:
        rgb = c.rgb
        if not rgb:
            return None
        if len(rgb) == 8:  # AARRGGBB
            return f"#{rgb[2:]}"
        if len(rgb) == 6:
            return f"#{rgb}"
    except Exception:
        pass
    return None

def _is_dark(hex_color: str) -> bool:
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        return False
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return lum < 80

def _border_css(b) -> str:
    if not b:
        return ""
    parts = []
    for side_name in ("left", "right", "top", "bottom"):
        side = getattr(b, side_name, None)
        if side and side.style:
            width = "1px"; style = "solid"
            if side.style in ("medium", "thick"): width = "2px"
            if side.style in ("hair", "dotted", "dashDot"): style = "dotted"
            color = _hex_color(getattr(side, "color", None)) or "#BEBEBE"
            parts.append(f"border-{side_name}:{width} {style} {color};")
    return "".join(parts)

def _align_css(al):
    if not al:
        return ""
    parts = []
    if al.horizontal: parts.append(f"text-align:{al.horizontal};")
    if al.vertical:   parts.append(f"vertical-align:{ {'center':'middle'}.get(al.vertical, al.vertical) };")
    parts.append("white-space:normal;" if getattr(al, "wrap_text", False) else "white-space:nowrap;")
    return "".join(parts)

def _font_css(f, force_white: bool=False):
    parts = []
    if f and f.bold:   parts.append("font-weight:700;")
    if f and f.italic: parts.append("font-style:italic;")
    if f and f.size:   parts.append(f"font-size:{f.size}px;")
    if force_white:
        parts.append("color:#ffffff;")
    else:
        col = _hex_color(getattr(f, "color", None))
        if col: parts.append(f"color:{col};")
    if not parts: parts.append("font-size:13px;")
    return "".join(parts)

def _fill_css(fill, fallback_blue=False):
    try:
        col = _hex_color(fill.fgColor) if fill and fill.fgColor else None
        if fallback_blue:  # titres/fonds sombres → bleu lisible
            return "background-color:#2c5282;"
        if col:
            return f"background-color:{col};"
    except Exception:
        pass
    return ""

# ---------- helpers sheet ----------
def _is_col_hidden(ws, idx: int) -> bool:
    cd = ws.column_dimensions.get(get_column_letter(idx))
    return bool(cd and cd.hidden)

def _is_row_hidden(ws, idx: int) -> bool:
    rd = ws.row_dimensions.get(idx)
    return bool(rd and rd.hidden)

def _is_empty(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")

def _detect_right_cut(ws, min_r, max_r, min_c, max_c,
                      look_rows: int = 120, empty_run_needed: int = 5) -> int:
    """
    Coupe à la première séquence d'au moins `empty_run_needed` colonnes vides consécutives,
    en n'observant que les `look_rows` premières lignes (le cœur des tableaux).
    """
    look_to_row = min(max_r, min_r + look_rows - 1)
    last_val_col = min_c
    empty_run = 0
    for c in range(min_c, max_c + 1):
        if _is_col_hidden(ws, c):
            continue
        has_val = False
        for r in range(min_r, look_to_row + 1):
            if _is_row_hidden(ws, r): 
                continue
            if not _is_empty(ws.cell(r, c).value):
                has_val = True
                break
        if has_val:
            last_val_col = c
            empty_run = 0
        else:
            empty_run += 1
            if empty_run >= empty_run_needed:
                break
    return last_val_col

def _visible_bounds(ws, col_cap: Optional[int] = None):
    """
    Borne visible + coupe zone commentaires à droite.
    `col_cap` permet d'imposer un nombre max de colonnes (UI).
    """
    max_r = ws.max_row
    max_c = ws.max_column

    min_r = 1
    while min_r <= max_r and _is_row_hidden(ws, min_r): min_r += 1
    while max_r >= 1 and _is_row_hidden(ws, max_r):     max_r -= 1

    min_c = 1
    while min_c <= max_c and _is_col_hidden(ws, min_c): min_c += 1

    # Détection du “mur noir” à droite
    detected_end = _detect_right_cut(ws, min_r, max_r, min_c, max_c,
                                     look_rows=120, empty_run_needed=5)
    max_c = detected_end

    # Forcer un plafond dur (ex. 15 colonnes max)
    max_c = min(max_c, 18)

    # Cap manuel (UI)
    if isinstance(col_cap, int) and col_cap > 0:
        max_c = min(max_c, min_c + col_cap - 1)


def _collect_merges(ws) -> Dict[Tuple[int, int], Merge]:
    merges: Dict[Tuple[int,int], Merge] = {}
    for rng in ws.merged_cells.ranges:
        min_c, min_r, max_c, max_r = range_boundaries(str(rng))
        if any(_is_row_hidden(ws, r) for r in range(min_r, max_r+1)):  continue
        if any(_is_col_hidden(ws, c) for c in range(min_c, max_c+1)):  continue
        merges[(min_r, min_c)] = Merge(min_c, min_r, max_c, max_r)
    return merges

def _extract_dataframe(ws, col_cap: Optional[int]) -> pd.DataFrame:
    min_r, max_r, min_c, max_c = _visible_bounds(ws, col_cap)
    data = []
    for r in range(min_r, max_r+1):
        if _is_row_hidden(ws, r): continue
        row_vals = []
        for c in range(min_c, max_c+1):
            if _is_col_hidden(ws, c): continue
            row_vals.append(ws.cell(r, c).value)
        data.append(row_vals)
    cols = [get_column_letter(c) for c in range(min_c, max_c+1) if not _is_col_hidden(ws, c)]
    return pd.DataFrame(data, columns=cols)

def _sheet_to_html(ws, col_cap: Optional[int]) -> str:
    min_r, max_r, min_c, max_c = _visible_bounds(ws, col_cap)
    merges = _collect_merges(ws)

    html = [
        '<div style="overflow:auto; max-height:80vh; border:1px solid #ddd; padding:8px;">',
        """
        <style>
        .arbs-table { border-collapse:collapse; font-family:Inter,system-ui,Arial; font-size:13px; }
        .arbs-table tr:nth-child(even) td { background-color:#fafafa; }  /* zebra rows */
        .arbs-table td { padding:4px 8px; }
        </style>
        """,
        '<table class="arbs-table">'
    ]

    covered = set()
    for (tl_r, tl_c), m in merges.items():
        for r in range(m.min_row, m.max_row+1):
            for c in range(m.min_col, m.max_col+1):
                if not (r == tl_r and c == tl_c):
                    covered.add((r, c))

    for r in range(min_r, max_r+1):
        if _is_row_hidden(ws, r): continue
        html.append("<tr>")
        for c in range(min_c, max_c+1):
            if _is_col_hidden(ws, c) or (r, c) in covered: continue

            cell = ws.cell(r, c)
            m = merges.get((r, c))
            rs = (m.max_row - m.min_row + 1) if m else 1
            cs = (m.max_col - m.min_col + 1) if m else 1

            raw_bg = _hex_color(getattr(getattr(cell, "fill", None), "fgColor", None))
            dark_bg = _is_dark(raw_bg) if raw_bg else False
            use_blue = bool(getattr(cell.font, "bold", False)) or dark_bg

            styles = []
            styles.append(_border_css(cell.border))
            styles.append(_align_css(cell.alignment))
            styles.append(_fill_css(cell.fill, fallback_blue=use_blue))
            styles.append(_font_css(cell.font, force_white=(use_blue or dark_bg)))
            if not cell.border or not any(getattr(cell.border, s).style for s in ("left","right","top","bottom")):
                styles.append("border:1px solid #e6e6e6;")

            val = cell.value
            value = "" if _is_empty(val) else str(val)
            value = re.sub(r"\n", "<br>", value)

            html.append(f'<td rowspan="{rs}" colspan="{cs}" style="{"".join(styles)}">{value}</td>')
        html.append("</tr>")
    html.append("</table></div>")
    return "".join(html)

# ===================== STREAMLIT TAB =====================
def render():
    st.header("Arbs")

    path = st.text_input("Chemin du fichier Excel des arbs :", DEFAULT_XLSX_PATH)
    col1, col2, col3 = st.columns([1,1,1], gap="small")
    editable   = col1.toggle("Mode éditable (grid)", value=False)
    show_export= col2.toggle("Activer export XLSX", value=True)
    # Limiteur manuel de colonnes (0 = auto)
    col_cap    = col3.number_input("Max colonnes à afficher (0 = auto)", min_value=0, max_value=200, value=0, step=1)

    try:
        wb = load_workbook(path, data_only=True)
    except Exception as e:
        st.error(f"Impossible d’ouvrir le fichier : {e}")
        st.stop()

    visible_sheets = [ws.title for ws in wb.worksheets if ws.sheet_state == "visible"]
    if not visible_sheets:
        st.warning("Aucune feuille visible dans ce classeur.")
        st.stop()

    sheet = st.selectbox("Feuille source :", visible_sheets,
                         index=visible_sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in visible_sheets else 0)
    ws = wb[sheet]
    cap = int(col_cap) if col_cap and col_cap > 0 else None

    if editable:
        st.caption("Vue éditable (valeurs uniquement).")
        df = _extract_dataframe(ws, cap)
        edited = st.data_editor(df, use_container_width=True, height=650)
        if show_export:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                edited.to_excel(writer, sheet_name="Arbs", index=False)
            st.download_button("Exporter la vue éditée en XLSX", data=out.getvalue(),
                               file_name="Arbs_export.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Rendu fidèle d’Excel (fusions, styles, couleurs plus visibles).")
        html = _sheet_to_html(ws, cap)
        st.markdown(html, unsafe_allow_html=True)

        if show_export:
            df = _extract_dataframe(ws, cap)
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Arbs_visible", index=False)
            st.download_button("Exporter la zone visible en XLSX", data=out.getvalue(),
                               file_name="Arbs_visible.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    import streamlit.web.bootstrap
    def main():
        render()
    streamlit.web.bootstrap.run(main, "", [], flag_options={})
