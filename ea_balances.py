# -------- ea_balances.py (robuste Fig.10) --------
# -*- coding: utf-8 -*-
import os, re, glob, platform
from pathlib import Path
import pandas as pd
import pdfplumber
import plotly.graph_objects as go

# ↑ Incrémente quand tu modifies ce fichier (le cache de l'app en tiendra compte)
PARSER_VERSION = "ea_parser_v9"

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
    Fig.10 parser robuste :
      - Recolle libellés et chiffres éclatés ('1 2 8' -> '128', '( 5 3 )' -> '(53)').
      - Découpe la page en 4 blocs (ancres pays; sinon 4 x 'Demand').
      - Après chaque label (Inland/Bunkers/Ref Supply/Blending), prend les 8 premiers
        tokens numériques en ignorant les marqueurs de trimestre (Q1 '24, etc.).
    """
    import re
    import pdfplumber

    countries = ["Netherlands", "Belgium", "Italy", "Total"]

    # ----- helpers -----
    def _parse_tok_loose(tok: str) -> int:
        t = tok.replace(" ", "").replace(",", "")
        if t == "--":
            return 0
        neg = False
        if t.startswith("(") and t.endswith(")"):
            neg = True
            t = t[1:-1]
        val = int(re.sub(r"[^\d-]", "", t) or "0")
        return -val if neg or t.startswith("-") else val

    def _grab8(chunk: str) -> list[int]:
        """1) vire Q1/Q2/'24  2) récupère 8 nombres/--  3) parse int"""
        c = re.sub(r"Q\s*\d\s*'?\d{2}", " ", chunk, flags=re.I)
        toks = re.findall(r"\(\s*-?\s*[\d,\s]+\s*\)|-?\d[\d,]*|--", c)
        vals = [_parse_tok_loose(t) for t in toks[:8]]
        return vals + [0] * (8 - len(vals))

    def _fix_tokens(txt: str) -> str:
        txt = (txt or "").replace("\u00A0", " ")
        txt = re.sub(r"[ \t]+", " ", txt)
        def squash(token: str, repl: str | None = None) -> str:
            letters = [c for c in token if c.isalpha()]
            if not letters: return txt
            pat = r'(?<![A-Za-z])' + r'[\s\.]*'.join(map(re.escape, letters)) + r'(?![A-Za-z])'
            return re.sub(pat, repl or "".join(letters), txt, flags=re.I)
        for t in ["Netherlands", "Belgium", "Italy", "Total",
                  "Demand", "Supply", "Inland", "Bunkers",
                  "Blending", "HSFO", "LSFO"]:
            txt = squash(t)
        txt = squash("Ref. Supply", "Ref Supply")
        txt = squash("Ref Supply", "Ref Supply")
        # recoller chiffres
        txt = re.sub(r'(?<=\d)\s+(?=\d)', '', txt)
        txt = re.sub(r'\(\s+', '(', txt); txt = re.sub(r'\s+\)', ')', txt)
        txt = re.sub(r'-\s+', '-', txt);  txt = re.sub(r'\s*,\s*', ',', txt)
        return txt

    def _section_slice(block: str, section: str) -> str:
        if section == "Demand":
            m = re.search(r"Demand\s+(.*?)(?:\bSupply\b)", block, flags=re.S)
        else:
            m = re.search(r"\bSupply\b\s*(.*)$", block, flags=re.S)
        if not m: raise RuntimeError(f"Section '{section}' introuvable.")
        return m.group(1)

    def _demand_grade_slice(demand_text: str, grade: str) -> str:
        other = "LSFO" if grade == "HSFO" else "HSFO"
        m0 = re.search(rf"\b{grade}\b", demand_text)
        if not m0: raise RuntimeError(f"Grade '{grade}' introuvable dans Demand.")
        tail = demand_text[m0.end():]
        m1 = re.search(rf"\b{other}\b", tail)
        return tail[:m1.start()] if m1 else tail

    def _grab_after_label(text: str, label: str) -> list[int]:
        m = re.search(rf"\b{label}\b", text)
        if not m: return [0]*8
        rest = text[m.end():]
        mstop = re.search(r"\b(Bunkers|Inland|Blending|Ref Supply|Supply|HSFO|LSFO|Netherlands|Belgium|Italy|Total)\b",
                          rest)
        chunk = rest[:mstop.start()] if mstop else rest
        return _grab8(chunk)

    def _extract_demand_parts(block: str, grade: str):
        dem = _section_slice(block, "Demand")
        seg = _demand_grade_slice(dem, grade)
        inland  = _grab_after_label(seg, "Inland")
        bunkers = _grab_after_label(seg, "Bunkers")
        return inland, bunkers

    def _extract_supply_parts(block: str, grade: str):
        sup = _section_slice(block, "Supply")
        ref_vals = [0]*8
        mref = re.search(r"\bRef\s+Supply\b(.*?)(?:\bBlending\b|\Z)", sup, flags=re.S)
        if mref: ref_vals = _grab8(mref.group(1))
        blend_vals = [0]*8
        mbl = re.search(r"\bBlending\b(.*)$", sup, flags=re.S)
        if mbl: blend_vals = _grab8(mbl.group(1))
        if all(v==0 for v in ref_vals) and all(v==0 for v in blend_vals):
            mdir = re.search(rf"\b{grade}\b(.*?)(?:\bHSFO\b|\bLSFO\b|\Z)", sup, flags=re.S)
            if mdir:
                ref_vals = _grab8(mdir.group(1)); blend_vals = [0]*8
        return ref_vals, blend_vals

    def _extract_country_balance_row(page_text: str, country: str) -> list[int]:
        m = re.search(rf"{re.escape(country)}\s+(.*)$", page_text, flags=re.M)
        return _grab8(m.group(1)) if m else [0]*8

    # ----- 1) localiser la page comme l'e-mail -----
    target = None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            if "Fig 10: Europe fuel oil balance by grade" in raw:
                target = _fix_tokens(raw); break
    if not target:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                if re.search(r"Fig\s*10:\s*Europe\s+fuel\s+oil\s+balance", raw, flags=re.I):
                    target = _fix_tokens(raw); break
    if not target:
        raise RuntimeError("Fig.10 not found in the PDF (email method + fixes).")

    # ----- 2) découpe en 4 blocs -----
    patterns: dict[str, str] = {}
    for c in countries:
        nxt = [k for k in countries if k != c]
        end_pat = r"(?:" + r"|".join(map(re.escape, nxt + ["Source:"])) + r")"
        m = re.search(rf"{re.escape(c)}[\s\S]*?(?={end_pat})", target)
        if m: patterns[c] = m.group(0)
        else:
            mt = re.search(rf"{re.escape(c)}[\s\S]*$", target)
            if mt: patterns[c] = mt.group(0)
    if len(patterns) < 4:
        starts = [m.start() for m in re.finditer(r"\bDemand\b", target)]
        if len(starts) >= 4:
            starts = starts[:4] + [len(target)]
            for i in range(4):
                patterns[countries[i]] = target[starts[i]:starts[i+1]]
        else:
            raise RuntimeError("Fig.10: cannot split page (no anchors, <4 'Demand').")

    # ----- 3) extraction numérique -----
    data: dict[str, dict] = {}
    for c, raw_block in patterns.items():
        block = _fix_tokens(raw_block)
        d = {"Balance_total": _extract_country_balance_row(target, c),
             "Demand": {}, "Supply": {}, "Demand_parts": {}, "Supply_parts": {}}
        for grade in ["HSFO", "LSFO"]:
            inland, bunkers = _extract_demand_parts(block, grade)
            d["Demand_parts"][grade] = {"Inland": inland, "Bunkers": bunkers}
            d["Demand"][grade] = [inland[i] + bunkers[i] for i in range(8)]
            ref_s, blend_s = _extract_supply_parts(block, grade)
            d["Supply_parts"][grade] = {"Ref": ref_s, "Blend": blend_s}
            d["Supply"][grade] = [ref_s[i] + blend_s[i] for i in range(8)]
            d[f"Balance_{grade}"] = [(ref_s[i] + blend_s[i]) - (inland[i] + bunkers[i]) for i in range(8)]
        data[c] = d
    return data


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
