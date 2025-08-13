# -------- ea_balances.py (robuste Fig.10) --------
# -*- coding: utf-8 -*-
import os, re, glob, platform
from pathlib import Path
import pandas as pd
import pdfplumber
import plotly.graph_objects as go

# ↑ Incrémente quand tu modifies ce fichier (le cache de l'app en tiendra compte)
PARSER_VERSION = "ea_parser_v11"

# ---------- Chemins ----------
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DEFAULT = REPO_ROOT / "EA balances"   # dossier du repo (Linux-friendly)

EA_ENV = os.getenv("EA_PDF_DIR")
if EA_ENV:
    BASE_PDF_DIR = Path(EA_ENV)
else:
    if platform.system() == "Windows":
        BASE_PDF_DIR = Path(r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\EA balances")
    else:
        BASE_PDF_DIR = LOCAL_DEFAULT

# Protection: sous Linux/Mac, si l'env pointe par erreur sur un UNC, rebasculer en local
if platform.system() != "Windows":
    s = str(BASE_PDF_DIR)
    if s.startswith("\\") or s.startswith("//"):
        BASE_PDF_DIR = LOCAL_DEFAULT

COUNTRIES = ["Netherlands", "Belgium", "Italy", "Total"]
QUARTERS  = ["Q1", "Q2", "Q3", "Q4"]

_tok  = r"(\(?-?\d[\d,]*\)?|--)"
_tok8 = rf"{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}"

# ---------- Utils texte ----------
def _norm_text(s: str) -> str:
    s = s.replace("\u00A0", " ")  # NBSP
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s

def _collapse_spaced_words(s: str) -> str:
    """
    Recolle 'N e t h e r l a n d s' -> 'Netherlands' (sans coller deux mots distincts).
    Recolle aussi D e m a n d / S u p p l y / I n l a n d / B u n k e r s.
    """
    return re.sub(
        r'(?<![A-Za-z])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])',
        lambda m: m.group(0).replace(" ", ""),
        s
    )

# ---------- Parsing helpers ----------
def _parse_tok(x: str) -> int:
    x = x.strip()
    if x == "--": return 0
    x = x.replace(",", "")
    if x.startswith("(") and x.endswith(")"): return -int(x[1:-1])
    return int(x)

def _section_slice(block: str, section: str) -> str:
    """Retourne uniquement la section Demand ou Supply du bloc pays (tolérant aux collages)."""
    if section == "Demand":
        m = re.search(r"Demand\s+(.*?)(?:\bSupply\b)", block, flags=re.S | re.I)
    else:
        m = re.search(r"\bSupply\b\s*(.*)$", block, flags=re.S | re.I)
    if not m:
        raise RuntimeError(f"Section '{section}' introuvable.")
    return m.group(1)

def _demand_grade_slice(demand_text: str, grade: str) -> str:
    """Isole la partie du grade dans Demand (tolère HSFOInland / LSFOInland)."""
    other = "LSFO" if grade == "HSFO" else "HSFO"
    pat = rf"(?<![A-Za-z]){grade}(?:(?![a-z])|(?=[A-Z]))"
    m_start = re.search(pat, demand_text)
    if not m_start:
        raise RuntimeError(f"Grade '{grade}' introuvable dans Demand.")
    tail = demand_text[m_start.end():]
    m_end = re.search(rf"(?<![A-Za-z]){other}(?:(?![a-z])|(?=[A-Z]))", tail)
    return tail[:m_end.start()] if m_end else tail

def _extract_subline(section_text: str, label: str) -> list[int]:
    m = re.search(rf"\b{label}\b\s+{_tok8}", section_text, flags=re.I)
    if not m: return [0]*8
    return [_parse_tok(v) for v in m.groups()]

def _extract_demand_parts(country_block: str, grade: str):
    dem = _section_slice(country_block, "Demand")
    seg = _demand_grade_slice(dem, grade)
    inland  = _extract_subline(seg, "Inland")
    bunkers = _extract_subline(seg, "Bunkers")
    return inland, bunkers

def _extract_supply_parts(country_block: str, grade: str):
    sup = _section_slice(country_block, "Supply")

    # Ref. Supply {grade} (jusqu'à 'Blending' sans imposer \n)
    ref_vals = [0]*8
    m_ref_blk = re.search(r"Ref\.?\s*Supply\s+(.*?)(?:\bBlending\b|\Z)", sup, flags=re.S | re.I)
    if m_ref_blk:
        m_ref = re.search(rf"{grade}\b\s+{_tok8}", m_ref_blk.group(1), flags=re.I)
        if m_ref:
            ref_vals = [_parse_tok(v) for v in m_ref.groups()]

    # Blending {grade}
    blend_vals = [0]*8
    m_bl_blk = re.search(r"\bBlending\b\s+(.*)$", sup, flags=re.S | re.I)
    if m_bl_blk:
        m_bl = re.search(rf"{grade}\b\s+{_tok8}", m_bl_blk.group(1), flags=re.I)
        if m_bl:
            blend_vals = [_parse_tok(v) for v in m_bl.groups()]

    # Fallback: ligne directe "HSFO ..."/"LSFO ..." dans Supply
    if all(v == 0 for v in ref_vals) and all(v == 0 for v in blend_vals):
        m_dir = re.search(rf"^\s*{grade}\b\s+{_tok8}", sup, flags=re.M | re.I)
        if m_dir:
            ref_vals = [_parse_tok(v) for v in m_dir.groups()]
            blend_vals = [0]*8

    return ref_vals, blend_vals

def _extract_country_balance_agg(page_text: str, country: str) -> list[int]:
    # Ne bloque pas l'exécution si non trouvé (met 0)
    m = re.search(rf"{re.escape(country)}\s+{_tok8}", page_text, flags=re.I)
    return [_parse_tok(v) for v in m.groups()] if m else [0]*8

# ---------- Fichiers ----------
def _get_latest_pdf_file() -> Path:
    if not BASE_PDF_DIR.exists():
        raise FileNotFoundError(
            "Répertoire introuvable pour EA PDFs : "
            f"{BASE_PDF_DIR}\n"
            f"OS={platform.system()} | cwd={Path.cwd()}\n"
            "En environnement Linux/Cloud, le partage UNC Windows n'est pas monté. "
            "Définis EA_PDF_DIR vers un dossier local accessible au runtime."
        )
    pdfs = glob.glob(str(BASE_PDF_DIR / "*.pdf"))
    if not pdfs:
        try:
            listing = "\n".join([p.name for p in sorted(BASE_PDF_DIR.iterdir())][:20])
        except Exception:
            listing = "(lecture du dossier impossible)"
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {BASE_PDF_DIR}\n"
            f"Contenu du dossier (extrait):\n{listing}"
        )
    latest = max(pdfs, key=os.path.getmtime)
    return Path(latest)

# ---------- Localisation fiable de la vraie page Fig.10 ----------
def _find_fig10_page_text(pdf_path: Path) -> tuple[str, int]:
    """
    Retourne (texte_de_la_page, page_number_1_based) pour la vraie Fig.10.
    Choisit la page avec le titre + le score max (Demand/Supply + pays),
    en ignorant la 'Table of figures'.
    """
    title_re = re.compile(r"Fig\.?\s*10\b.*Europe\s+fuel\s+oil\s+balance", re.I)
    country_re = re.compile(r"\b(Netherlands|Belgium|Italy|Total)\b", re.I)

    best_txt, best_score, best_page_no = None, -1, -1
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            # Texte "meilleur effort"
            page_txt = ""
            for xt in (10, 9, 8, 7, 6, 5, 4, 3, 2):
                try:
                    t = page.extract_text(x_tolerance=xt, y_tolerance=xt, layout=True) or ""
                except TypeError:
                    t = page.extract_text(layout=True) or page.extract_text() or ""
                t = _collapse_spaced_words(_norm_text(t))
                if len(t) > len(page_txt):
                    page_txt = t

            if not title_re.search(page_txt):
                continue
            if re.search(r"\bTable\s+of\s+figures\b", page_txt, re.I):
                continue

            score = len(re.findall(r"\bDemand\b", page_txt, re.I)) \
                    + len(re.findall(r"\bSupply\b", page_txt, re.I)) \
                    + 2 * len(country_re.findall(page_txt))  # pays comptent double

            if score > best_score:
                best_score, best_txt, best_page_no = score, page_txt, idx

    if best_txt is None:
        raise RuntimeError("Fig.10 introuvable: aucune page avec le titre (hors 'Table of figures').")

    return best_txt, best_page_no

def _parse_fig10(pdf_path: Path) -> dict:
    """
    Parser Fig.10 (Europe fuel oil balance by grade) – robuste à l'extraction PDF.
    - Localise la bonne page via _find_fig10_page_text (ignore la Table of figures).
    - Recolle libellés ET chiffres éclatés.
    - Découpe le texte par pays via ancres de début de ligne (^Netherlands, ^Belgium, ^Italy, ^Total).
    - Lit 8 valeurs après chaque libellé (Inland, Bunkers, Ref Supply -> HSFO/LSFO, Blending -> HSFO/LSFO).
    """
    import re

    # ---------- helpers communs ----------
    def _tok_to_int(tok: str) -> int:
        if tok is None:
            return 0
        t = tok.replace(" ", "").replace(",", "")
        if t == "--":
            return 0
        neg = False
        if t.startswith("(") and t.endswith(")"):
            neg = True
            t = t[1:-1]
        t = re.sub(r"[^\d-]", "", t) or "0"
        v = int(t)
        return -v if neg or tok.strip().startswith("-") else v

    NUM = r"\(\s*-?\s*[\d,\s]+\s*\)|-?\d[\d,\s]*|--"  # autorise espaces/parenthèses

    def _fix_tokens(txt: str) -> str:
        # normalisation + recollage des libellés et des chiffres
        txt = (txt or "").replace("\u00A0", " ")
        txt = txt.replace("–", "-").replace("—", "-")
        txt = re.sub(r"[ \t]+", " ", txt)

        def squash(token: str, repl: str | None = None) -> str:
            letters = [c for c in token if c.isalpha()]
            if not letters:
                return txt
            pat = r'(?m)(?<![A-Za-z])' + r'[\s\.]*'.join(map(re.escape, letters)) + r'(?![A-Za-z])'
            return re.sub(pat, repl or "".join(letters), txt, flags=re.I)

        for t in ["Netherlands", "Belgium", "Italy", "Total",
                  "Demand", "Supply", "Ref. Supply", "Ref Supply",
                  "Blending", "Inland", "Bunkers", "HSFO", "LSFO"]:
            txt = squash(t, "Ref Supply" if t in ("Ref. Supply", "Ref Supply") else None)

        # recoller les chiffres éclatés
        txt = re.sub(r'(?<=\d)\s+(?=\d)', '', txt)  # 1 2 8 -> 128
        txt = re.sub(r'\(\s+', '(', txt)
        txt = re.sub(r'\s+\)', ')', txt)
        txt = re.sub(r'-\s+', '-', txt)
        txt = re.sub(r'\s*,\s*', ',', txt)
        return txt

    def _grab8(chunk: str) -> list[int]:
        # supprime les intitulés de colonnes (Q1 '25 etc.)
        c = re.sub(r"Q\s*\d\s*'?\d{2}", " ", chunk, flags=re.I)
        vals = re.findall(NUM, c)
        vals = [_tok_to_int(t) for t in vals[:8]]
        return vals + [0]*(8-len(vals))

    def _slice_between(text: str, start_pat: str, end_pat: str) -> str:
        m0 = re.search(start_pat, text, flags=re.S|re.M|re.I)
        if not m0:
            return ""
        rest = text[m0.end():]
        m1 = re.search(end_pat, rest, flags=re.S|re.M|re.I)
        return rest[:m1.start()] if m1 else rest

    # ---------- 1) localiser la bonne page ----------
    page_text, _ = _find_fig10_page_text(pdf_path)  # ← utilise ta fonction existante
    page_text = _fix_tokens(page_text)

    # ---------- 2) découper en 4 blocs par pays (ancres de début de ligne) ----------
    countries = ["Netherlands", "Belgium", "Italy", "Total"]
    blocks: dict[str, str] = {}
    for i, c in enumerate(countries):
        start = rf"(?m)^{re.escape(c)}\b"
        if i < len(countries)-1:
            nxt = rf"(?m)^(?:{re.escape(countries[i+1])}|Source:)\b"
        else:
            nxt = rf"(?m)^Source:\b"
        blk = _slice_between(page_text, start, nxt)
        if not blk:
            # dernier recours: jusqu'à la fin si 'Source:' absent
            m = re.search(start, page_text, flags=re.M|re.I)
            if m:
                blocks[c] = page_text[m.start():]
        else:
            blocks[c] = c + "\n" + blk  # on garde le nom en tête de bloc

    if len(blocks) != 4:
        # on ne force plus le split par 'Demand' -> on échoue explicitement
        raise RuntimeError("Fig.10: anchors par pays introuvables (Netherlands/Belgium/Italy/Total).")

    # ---------- 3) extraction par bloc ----------
    out: dict[str, dict] = {}
    for c, blk in blocks.items():
        b = _fix_tokens(blk)

        # sections
        demand_sec = _slice_between(b, r"(?m)^Demand\b", r"(?m)^Supply\b")
        supply_sec = _slice_between(b, r"(?m)^Supply\b", r"(?m)^(?:Netherlands|Belgium|Italy|Total|Source:)\b")

        def demand_grade(grade: str) -> tuple[list[int], list[int]]:
            seg = _slice_between(demand_sec, rf"(?m)^{grade}\b", rf"(?m)^(?:HSFO|LSFO|Supply)\b")
            inland  = _grab8(_slice_between(seg, r"(?m)^Inland\b", r"(?m)^(?:Bunkers|HSFO|LSFO|Supply)\b"))
            bunkers = _grab8(_slice_between(seg, r"(?m)^Bunkers\b", r"(?m)^(?:Inland|HSFO|LSFO|Supply)\b"))
            return inland, bunkers

        def supply_grade(grade: str) -> tuple[list[int], list[int]]:
            ref_blk   = _slice_between(supply_sec, r"(?m)^Ref Supply\b", r"(?m)^Blending\b")
            blend_blk = _slice_between(supply_sec, r"(?m)^Blending\b", r"(?m)^(?:Netherlands|Belgium|Italy|Total|Source:)\b")
            ref_vals   = _grab8(_slice_between(ref_blk, rf"(?m)^{grade}\b", r"(?m)^(?:HSFO|LSFO|Blending|$)"))
            blend_vals = _grab8(_slice_between(blend_blk, rf"(?m)^{grade}\b", r"(?m)^(?:HSFO|LSFO|$)"))
            # fallback: parfois les grades sont directement listés sous Supply
            if all(v == 0 for v in ref_vals) and all(v == 0 for v in blend_vals):
                direct = _slice_between(supply_sec, rf"(?m)^{grade}\b", r"(?m)^(?:HSFO|LSFO|$)")
                ref_vals = _grab8(direct)
                blend_vals = [0]*8
            return ref_vals, blend_vals

        data_country = {
            "Balance_total": _grab8(_slice_between(b, rf"(?m)^{re.escape(c)}\b", r"(?m)^Demand\b")),
            "Demand": {}, "Supply": {},
            "Demand_parts": {}, "Supply_parts": {}
        }

        for grade in ("HSFO", "LSFO"):
            inl, bun = demand_grade(grade)
            data_country["Demand_parts"][grade] = {"Inland": inl, "Bunkers": bun}
            data_country["Demand"][grade] = [inl[i] + bun[i] for i in range(8)]

            ref_s, blend_s = supply_grade(grade)
            data_country["Supply_parts"][grade] = {"Ref": ref_s, "Blend": blend_s}
            data_country["Supply"][grade] = [ref_s[i] + blend_s[i] for i in range(8)]

            data_country[f"Balance_{grade}"] = [
                (ref_s[i] + blend_s[i]) - (inl[i] + bun[i]) for i in range(8)
            ]

        out[c] = data_country

    return out


def _to_tidy_dataframe(parsed: dict) -> pd.DataFrame:
    rows = []
    for c in COUNTRIES:
        for metric in ["Demand", "Supply"]:
            for grade in ["HSFO", "LSFO"]:
                for i, val in enumerate(parsed[c][metric][grade]):
                    rows.append(dict(
                        country=c, metric=metric, grade=grade,
                        year=2025 if i < 4 else 2026,
                        quarter=QUARTERS[i % 4], q_num=(i % 4) + 1, value=val
                    ))
        for grade in ["HSFO", "LSFO"]:
            for i, val in enumerate(parsed[c][f"Balance_{grade}"]):
                rows.append(dict(
                    country=c, metric="Balance", grade=grade,
                    year=2025 if i < 4 else 2026,
                    quarter=QUARTERS[i % 4], q_num=(i % 4) + 1, value=val
                ))
    return pd.DataFrame(rows)

# --------- PUBLIC: charge tout (Demand, Supply, Balance) ---------
def load_ea_data():
    """
    Retourne:
      data[metric][grade][year] = DataFrame index=Quarter(1..4), columns=COUNTRIES
      où metric ∈ {"Demand","Supply","Balance"} et grade ∈ {"HSFO","LSFO"}.
    """
    pdf_path = _get_latest_pdf_file()
    parsed = _parse_fig10(pdf_path)
    df = _to_tidy_dataframe(parsed)

    out = {m: {"HSFO": {}, "LSFO": {}} for m in ["Demand", "Supply", "Balance"]}
    for metric in ["Demand", "Supply", "Balance"]:
        for grade in ["HSFO", "LSFO"]:
            sub = df[(df["metric"] == metric) & (df["grade"] == grade)]
            for yr in [2025, 2026]:
                piv = sub[sub["year"] == yr].pivot_table(
                    index="q_num", columns="country", values="value"
                ).sort_index()
                piv.index.name = "Quarter"
                out[metric][grade][yr] = piv
    return out

# --------- PUBLIC: figures Plotly (par pays) ---------
def plot_ea(data, metric: str, grade: str):
    """
    data: résultat de load_ea_data()
    metric: "Demand" | "Supply" | "Balance"
    grade: "HSFO" | "LSFO"
    Retourne: dict {f"{metric} {grade} - {country}": go.Figure}
    """
    figs = {}
    piv25 = data[metric][grade].get(2025)
    piv26 = data[metric][grade].get(2026)
    for country in COUNTRIES:
        fig = go.Figure()
        if piv25 is not None and country in piv25.columns:
            fig.add_trace(go.Scatter(
                x=piv25.index, y=piv25[country],
                mode="lines+markers", name="2025", line=dict(color="black")
            ))
        if piv26 is not None and country in piv26.columns:
            fig.add_trace(go.Scatter(
                x=piv26.index, y=piv26[country],
                mode="lines+markers", name="2026", line=dict(color="red")
            ))
        fig.update_layout(
            title=f"{metric} {grade} - {country}",
            xaxis=dict(
                title="Quarter",
                tickmode="array",
                tickvals=[1,2,3,4],
                ticktext=["Q1","Q2","Q3","Q4"]
            ),
            yaxis_title="kb/d",
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        figs[f"{metric} {grade} - {country}"] = fig
    return figs
