# -------- Config (chemin configurable) --------
import os, re, glob, platform
from pathlib import Path
import pandas as pd
import pdfplumber
import plotly.graph_objects as go

# Dossier par défaut = dossier local du projet: "<repo>/EA balances"
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DEFAULT = REPO_ROOT / "EA balances"

# 1) Si EA_PDF_DIR est défini, on l'utilise tel quel
EA_ENV = os.getenv("EA_PDF_DIR")

# 2) Sinon: Windows -> UNC ; Linux/Mac -> dossier local dans le repo
if EA_ENV:
    BASE_PDF_DIR = Path(EA_ENV)
else:
    if platform.system() == "Windows":
        BASE_PDF_DIR = Path(r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\EA balances")
    else:
        BASE_PDF_DIR = LOCAL_DEFAULT

# Protection: si on n'est PAS sous Windows et que l'env pointe vers un UNC, on rebascule en local
if platform.system() != "Windows":
    s = str(BASE_PDF_DIR)
    if s.startswith("\\") or s.startswith("//"):
        BASE_PDF_DIR = LOCAL_DEFAULT

COUNTRIES = ["Netherlands", "Belgium", "Italy", "Total"]
QUARTERS  = ["Q1", "Q2", "Q3", "Q4"]

_tok  = r"(\(?-?\d[\d,]*\)?|--)"
_tok8 = rf"{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}\s+{_tok}"

def _parse_tok(x: str) -> int:
    x = x.strip()
    if x == "--": return 0
    x = x.replace(",", "")
    if x.startswith("(") and x.endswith(")"): return -int(x[1:-1])
    return int(x)

def _section_slice(block: str, section: str) -> str:
    """Retourne uniquement la section Demand ou Supply du bloc pays."""
    if section == "Demand":
        # pas de \n obligatoire avant Supply (souvent collé/inline)
        m = re.search(r"Demand\s+(.*?)(?:\bSupply\b)", block, flags=re.S | re.I)
    else:
        m = re.search(r"\bSupply\b\s*(.*)$", block, flags=re.S | re.I)
    if not m:
        raise RuntimeError(f"Section '{section}' introuvable.")
    return m.group(1)

def _demand_grade_slice(demand_text: str, grade: str) -> str:
    """Isole la partie du grade dans Demand (tolère HSFOInland, etc.)."""
    other = "LSFO" if grade == "HSFO" else "HSFO"
    pat = rf"(?<![A-Za-z]){grade}(?:(?![a-z])|(?=[A-Z]))"
    m_start = re.search(pat, demand_text)
    if not m_start:
        raise RuntimeError(f"Grade '{grade}' introuvable dans Demand.")
    tail = demand_text[m_start.end():]
    m_end = re.search(rf"(?<![A-Za-z]){other}(?:(?![a-z])|(?=[A-Z]))", tail)
    return tail[:m_end.start()] if m_end else tail

def _extract_subline(section_text: str, label: str) -> list[int]:
    m = re.search(rf"\b{label}\b\s+{_tok8}", section_text)
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
    m = re.search(rf"{re.escape(country)}\s+{_tok8}", page_text)
    if not m: raise RuntimeError(f"Balance agrégée introuvable pour {country}.")
    return [_parse_tok(v) for v in m.groups()]

def _get_latest_pdf_file() -> Path:
    # 1) le dossier existe ?
    if not BASE_PDF_DIR.exists():
        raise FileNotFoundError(
            "Répertoire introuvable pour EA PDFs : "
            f"{BASE_PDF_DIR}\n"
            f"OS={platform.system()} | cwd={Path.cwd()}\n"
            "En environnement Linux/Cloud, le partage UNC Windows n'est pas monté. "
            "Définis EA_PDF_DIR vers un dossier local accessible au runtime."
        )

    # 2) y a-t-il des PDF ?
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

def _norm_text(s: str) -> str:
    # Normalise espaces et ponctuations légères pour fiabiliser les regex
    s = s.replace("\u00A0", " ")  # espace insécable
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s

def _norm_text(s: str) -> str:
    # normalise espaces & ponctuation légère
    s = s.replace("\u00A0", " ")  # NBSP
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s

def _collapse_spaced_words(s: str) -> str:
    """
    Recolle les mots quand le PDF a 'N e t h e r l a n d s' -> 'Netherlands'
    et les libellés 'D e m a n d', 'S u p p l y', 'I n l a n d', 'B u n k e r', etc.
    NE TOUCHE PAS aux séparations entre mots.
    """
    # lettres espacées (>=2 lettres)
    s = re.sub(
        r'(?<![A-Za-z])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])',
        lambda m: m.group(0).replace(" ", ""),
        s
    )
    return s


def _parse_fig10(pdf_path: Path) -> dict:
    import pdfplumber

    def _norm_text(s: str) -> str:
        s = s.replace("\u00A0", " ")
        s = s.replace("–", "-").replace("—", "-")
        s = re.sub(r"[ \t]+", " ", s)
        return s

    def _collapse_spaced_words(s: str) -> str:
        # Recolle 'N e t h e r l a n d s' -> 'Netherlands' sans coller les mots entre eux
        return re.sub(
            r'(?<![A-Za-z])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])',
            lambda m: m.group(0).replace(" ", ""),
            s
        )

    # 1) Localiser la page Fig.10 (permissif)
    title_re = re.compile(r"Fig\.?\s*10\b.*Europe\s+fuel\s+oil\s+balance", re.I)
    target_page_text = None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            best = ""
            for xt in (8, 7, 6, 5, 4, 3, 2):
                try:
                    t = page.extract_text(x_tolerance=xt, y_tolerance=xt, layout=True) or ""
                except TypeError:
                    t = page.extract_text(layout=True) or page.extract_text() or ""
                t = _collapse_spaced_words(_norm_text(t))
                if len(t) > len(best):
                    best = t
            if title_re.search(best):
                target_page_text = best
                break

    if not target_page_text:
        raise RuntimeError("Fig.10 introuvable (titre non détecté).")

    # 2) Découpe en blocs pays — EXACTEMENT comme dans ton script email
    countries = ["Netherlands", "Belgium", "Italy", "Total"]
    patterns = {}
    for c in countries:
        next_keywords = [k for k in countries if k != c]
        end_pat = r"(?:" + r"|".join(map(re.escape, next_keywords + ["Source"])) + r")"
        m = re.search(rf"{re.escape(c)}[\s\S]*?(?={end_pat})", target_page_text, flags=re.I)
        if m:
            patterns[c] = m.group(0)
        else:
            # si c est le dernier (ex. 'Total' tout en bas), prends jusqu'à la fin
            m_tail = re.search(rf"{re.escape(c)}[\s\S]*$", target_page_text, flags=re.I)
            if m_tail:
                patterns[c] = m_tail.group(0)
            else:
                raise RuntimeError(f"Bloc pays introuvable: {c}")

    # 3) Extraction identique au script email (adapté aux collages)
    tok  = r"(\(?-?\d[\d,]*\)?|--)"
    tok8 = rf"{tok}\s+{tok}\s+{tok}\s+{tok}\s+{tok}\s+{tok}\s+{tok}\s+{tok}"

    def _parse_tok_local(x: str) -> int:
        x = x.strip()
        if x == "--": return 0
        x = x.replace(",", "")
        if x.startswith("(") and x.endswith(")"): return -int(x[1:-1])
        return int(x)

    def _extract_country_balance_agg_local(page_text: str, country: str) -> list[int]:
        # la ligne agrégée "Italy 64 57 ..." etc. (si manquante, on met 0)
        m = re.search(rf"{re.escape(country)}\s+{tok8}", page_text)
        return [_parse_tok_local(v) for v in m.groups()] if m else [0]*8

    data = {}
    for c, block in patterns.items():
        d = {}
        d["Balance_total"] = _extract_country_balance_agg_local(target_page_text, c)
        d["Demand"] = {}
        d["Supply"] = {}
        d["Demand_parts"] = {}
        d["Supply_parts"] = {}

        for grade in ["HSFO", "LSFO"]:
            # DEMAND
            dem = _section_slice(block, "Demand")
            seg = _demand_grade_slice(dem, grade)
            m_in  = re.search(rf"\bInland\b\s+{tok8}", seg, flags=re.I)
            m_bn  = re.search(rf"\bBunkers\b\s+{tok8}", seg, flags=re.I)
            inland  = [_parse_tok_local(v) for v in m_in.groups()]  if m_in  else [0]*8
            bunkers = [_parse_tok_local(v) for v in m_bn.groups()]  if m_bn  else [0]*8
            d["Demand_parts"][grade] = {"Inland": inland, "Bunkers": bunkers}
            d["Demand"][grade] = [inland[i] + bunkers[i] for i in range(8)]

            # SUPPLY (Ref. Supply + Blending)
            ref_s, blend_s = _extract_supply_parts(block, grade)
            d["Supply_parts"][grade] = {"Ref": ref_s, "Blend": blend_s}
            d["Supply"][grade] = [ref_s[i] + blend_s[i] for i in range(8)]

            # BALANCE(grade)
            d[f"Balance_{grade}"] = [
                (ref_s[i] + blend_s[i]) - (inland[i] + bunkers[i]) for i in range(8)
            ]

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
