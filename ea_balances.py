# -------- ea_balances.py (robuste Fig.10) --------
# -*- coding: utf-8 -*-
import os, re, glob, platform
from pathlib import Path
import pandas as pd
import pdfplumber
import plotly.graph_objects as go

# ↑ Incrémente quand tu modifies ce fichier (le cache de l'app en tiendra compte)
PARSER_VERSION = "ea_parser_v14"

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

# alias tolérés pour variations d’EA / extraction PDF
COUNTRY_ALIASES: dict[str, list[str]] = {
        "Netherlands": [r"Netherlands", r"The\s+Netherlands", r"\bNL\b"],
        "Belgium":     [r"Belgium", r"\bBE\b"],
        "Italy":       [r"Italy", r"\bIT\b"],
        "Total":       [r"Total(?:\s*Europe)?", r"Europe\s*Total", r"EU\s*Total"],
    }


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

    1) Parcourt toutes les pages qui contiennent le titre "Fig 10: Europe fuel oil balance"
       et IGNORE TOUTES les variantes de "Table of figures" (y compris lettres espacées).
    2) Score: Demand/Supply + pays -> on garde la meilleure.
    3) Fallback: on lit la page "Table of figures" pour extraire le numéro de page de Fig.10
       (ex: '... kb/d  9') et on ouvre directement cette page.
    """
    import pdfplumber, re

    def _clean(s: str) -> str:
        # même normalisation que ton code
        s = _collapse_spaced_words(_norm_text(s or ""))
        return s

    # titre Fig.10
    title_re = re.compile(r"Fig\.?\s*10\b.*Europe\s+fuel\s+oil\s+balance", re.I)
    # détection robuste du libellé "Table of figures" même si lettres espaçées
    tof_re = re.compile(r"t\s*a\s*b\s*l\s*e\s+o\s*f\s+f\s*i\s*g\s*u\s*r\s*e\s*s", re.I)

    countries = ("Netherlands", "Belgium", "Italy", "Total")

    best_txt, best_score, best_page_no = None, -1, -1

    with pdfplumber.open(pdf_path) as pdf:
        # --- 1) pages candidates avec le titre (en ignorant la table des figures) ---
        for idx, page in enumerate(pdf.pages, start=1):
            try:
                t = page.extract_text(x_tolerance=8, y_tolerance=8, layout=True) or ""
            except TypeError:
                t = page.extract_text() or ""
            t = _clean(t)

            if not title_re.search(t):
                continue
            if tof_re.search(t):  # ignorer *toutes* les variantes "Table of figures"
                continue

            score = (
                len(re.findall(r"\bDemand\b", t, re.I))
                + len(re.findall(r"\bSupply\b", t, re.I))
                + 2 * sum(bool(re.search(rf"\b{c}\b", t, re.I)) for c in countries)
            )

            if score > best_score:
                best_txt, best_score, best_page_no = t, score, idx

        if best_txt:
            return best_txt, best_page_no

        # --- 2) Fallback: prendre le numéro dans la Table of figures ---
        tof_page_no = None
        tof_text = ""
        for idx, page in enumerate(pdf.pages, start=1):
            t = _clean(page.extract_text() or "")
            if tof_re.search(t):
                tof_text = t
                m = re.search(r"Fig\s*10:.*?(\d{1,3})\s*$", t, re.I | re.M)
                if m:
                    tof_page_no = int(m.group(1))
                break

        if tof_page_no and 1 <= tof_page_no <= len(pdf.pages):
            # ouvrir directement la page annoncée (ex: 9)
            try:
                t = pdf.pages[tof_page_no - 1].extract_text(x_tolerance=8, y_tolerance=8, layout=True) or ""
            except TypeError:
                t = pdf.pages[tof_page_no - 1].extract_text() or ""
            return _clean(t), tof_page_no

    raise RuntimeError("Fig.10 introuvable: ni page avec le titre, ni Table of figures exploitable.")


def _parse_fig10(pdf_path: Path) -> dict:
    """
    Parser Fig.10 (Europe fuel oil balance by grade) – robuste à l'extraction PDF.
    - Localise la bonne page via _find_fig10_page_text (ignore la Table of figures).
    - Recolle libellés ET chiffres éclatés.
    - Découpe le texte par pays (alias: The Netherlands / Europe Total, etc.).
    - Accepte VLSFO comme alias de LSFO, et diverses variantes des libellés.
    """
    import re

    # ---------- helpers communs ----------
    NUM = r"\(\s*[-–]?\s*[\d\.,\s]+\s*\)|[-–]?\d[\d\.,\s]*|--"  # nombres, parens négatives, tirets unicode

    def _tok_to_int(tok: str) -> int:
        if not tok:
            return 0
        t = tok.replace(" ", "").replace("\u00A0", "")
        t = t.replace("–", "-").replace("—", "-")
        if t == "--":
            return 0
        neg = False
        if t.startswith("(") and t.endswith(")"):
            neg = True
            t = t[1:-1]
        # normaliser séparateurs , .
        t = t.replace(",", "")
        t = re.sub(r"[^\d\-]", "", t) or "0"
        v = int(t)
        return -v if neg or tok.strip().startswith(("-", "–", "—")) else v

    def _clean_and_repair(txt: str) -> str:
        # normalisation + recollage
        txt = (txt or "").replace("\u00A0", " ")
        txt = txt.replace("–", "-").replace("—", "-")
        txt = re.sub(r"[ \t]+", " ", txt)

        # recoller mots espacés (N e t h e r l a n d s, D e m a n d, etc.)
        txt = _collapse_spaced_words(txt)

        # harmoniser "Ref. Supply" -> "Refinery Supply"
        txt = re.sub(r"\bRef\.?\s*Supply\b", "Refinery Supply", txt, flags=re.I)

        # normaliser les grades: LSFO / VLSFO => LSFO; HSFO inchangé
        txt = re.sub(r"\bVLSFO\b", "LSFO", txt, flags=re.I)

        # recoller chiffres éclatés
        txt = re.sub(r'(?<=\d)\s+(?=\d)', '', txt)  # 1 2 8 -> 128
        txt = re.sub(r'\(\s+', '(', txt)
        txt = re.sub(r'\s+\)', ')', txt)
        txt = re.sub(r'-\s+', '-', txt)
        txt = re.sub(r'\s*,\s*', ',', txt)
        return txt

    def _grab8(chunk: str) -> list[int]:
        # retirer éventuelles en-têtes de colonnes (Q1 '24, Q2 '25, etc.)
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
    page_text, page_no = _find_fig10_page_text(pdf_path)
    page_text = _clean_and_repair(page_text)

    # ---------- 2) découper en 4 blocs par pays ----------
    countries = ["Netherlands", "Belgium", "Italy", "Total"]
    COUNTRY_ALIASES: dict[str, list[str]] = {
        "Netherlands": [r"Netherlands", r"The\s+Netherlands", r"\bNL\b"],
        "Belgium":     [r"Belgium", r"\bBE\b"],
        "Italy":       [r"Italy", r"\bIT\b"],
        "Total":       [r"Total(?:\s*Europe)?", r"Europe\s*Total", r"EU\s*Total"],
    }

    # positions trouvées
    hits = []
    for c in countries:
        pos = None
        for pat in COUNTRY_ALIASES[c]:
            m = re.search(rf"(?<![A-Za-z]){pat}(?![A-Za-z])", page_text, flags=re.I)
            if m:
                pos = m.start()
                break
        if pos is not None:
            hits.append((c, pos))

    blocks: dict[str, str] = {}
    if len(hits) >= 2:
        hits.sort(key=lambda kv: kv[1])
        for i, (c, pos) in enumerate(hits):
            end = hits[i+1][1] if i+1 < len(hits) else len(page_text)
            blocks[c] = page_text[pos:end]

    # fallback: découpe grossière par 4 "Demand"
    if len(blocks) < 4:
        spans = [m.start() for m in re.finditer(r"\bDemand\b", page_text, flags=re.I)]
        if len(spans) >= 4:
            spans = spans[:4] + [len(page_text)]
            for i in range(4):
                blocks[countries[i]] = page_text[spans[i]:spans[i+1]]

    # en dernier recours: blocs vides pour ceux manquants
    for c in countries:
        blocks.setdefault(c, "")

    # ---------- 3) alias de libellés (Demand/Supply/parts) ----------
    # on prépare des patrons larges pour les variantes fréquentes dans EA
    PAT = {
        "demand":        r"\bDemand\b",
        "supply":        r"\bSupply\b",
        "inland":        r"\b(?:Inland|Domestic)\b",
        "bunkers":       r"\b(?:Bunkers|Marine\s+Bunkers|Marine)\b",
        "ref_supply":    r"\b(?:Refinery\s*Supply|Refinery\s*Output|Ref\s*Supply)\b",
        "blending":      r"\b(?:Blending|Blend)\b",
        "grade_sep":     r"\b(?:HSFO|LSFO)\b",   # LSFO inclut VLSFO via normalisation
    }

    out: dict[str, dict] = {}

    for c, blk in blocks.items():
        b = _clean_and_repair(blk)

        # sections
        demand_sec = _slice_between(b, PAT["demand"], PAT["supply"])
        supply_sec = _slice_between(b, PAT["supply"], r"(?:Netherlands|Belgium|Italy|Total|Source:)\b")

        def demand_grade(grade: str) -> tuple[list[int], list[int]]:
            # isole le segment du grade dans Demand jusqu’au prochain grade ou Supply
            seg = _slice_between(demand_sec, rf"\b{grade}\b", rf"(?:{PAT['grade_sep']}|{PAT['supply']})")
            inland  = _grab8(_slice_between(seg, PAT["inland"],  rf"(?:{PAT['bunkers']}|{PAT['grade_sep']}|{PAT['supply']})"))
            bunkers = _grab8(_slice_between(seg, PAT["bunkers"], rf"(?:{PAT['inland']}|{PAT['grade_sep']}|{PAT['supply']})"))
            return inland, bunkers

        def supply_grade(grade: str) -> tuple[list[int], list[int]]:
            ref_blk   = _slice_between(supply_sec, PAT["ref_supply"], PAT["blending"])
            blend_blk = _slice_between(supply_sec, PAT["blending"],   r"(?:Netherlands|Belgium|Italy|Total|Source:)\b")
            ref_vals   = _grab8(_slice_between(ref_blk,   rf"\b{grade}\b", rf"(?:{PAT['grade_sep']}|{PAT['blending']}|$)"))
            blend_vals = _grab8(_slice_between(blend_blk, rf"\b{grade}\b", rf"(?:{PAT['grade_sep']}|$)"))
            # fallback: lignes directes "HSFO ..."/"LSFO ..." dans Supply
            if all(v == 0 for v in ref_vals) and all(v == 0 for v in blend_vals):
                direct = _slice_between(supply_sec, rf"\b{grade}\b", rf"(?:{PAT['grade_sep']}|$)")
                ref_vals = _grab8(direct); blend_vals = [0]*8
            return ref_vals, blend_vals

        data_country = {
            "Balance_total": _grab8(_slice_between(b, rf"{re.escape(c)}\b", PAT["demand"])),
            "Demand": {}, "Supply": {},
            "Demand_parts": {}, "Supply_parts": {}
        }

        for grade in ("HSFO", "LSFO"):  # LSFO couvre aussi VLSFO (normalisé)
            inl, bun = demand_grade(grade)
            data_country["Demand_parts"][grade] = {"Inland": inl, "Bunkers": bun}
            data_country["Demand"][grade] = [inl[i] + bun[i] for i in range(8)]

            ref_s, blend_s = supply_grade(grade)
            data_country["Supply_parts"][grade] = {"Ref": ref_s, "Blend": blend_s}
            data_country["Supply"][grade] = [ref_s[i] + blend_s[i] for i in range(8)]

            data_country[f"Balance_{grade}"] = [
                (ref_s[i] + blend_s[i]) - (inl[i] + bun[i]) for i in range(8)
            ]

        # petit log pour diagnostiquer si tout est à zéro
        if all(v == 0 for v in data_country["Demand"]["HSFO"] + data_country["Demand"]["LSFO"]):
            print(f"[EA parser] Alerte: aucune valeur Demand non nulle pour {c} (page {page_no}).")
        if all(v == 0 for v in data_country["Supply"]["HSFO"] + data_country["Supply"]["LSFO"]):
            print(f"[EA parser] Alerte: aucune valeur Supply non nulle pour {c} (page {page_no}).")

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
