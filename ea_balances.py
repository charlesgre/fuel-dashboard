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
    if section == "Demand":
        m = re.search(r"Demand\s+(.*?)(?:\n\s*Supply\b)", block, flags=re.S)
    else:
        m = re.search(r"\bSupply\s+(.*)$", block, flags=re.S)
    if not m: raise RuntimeError(f"Section '{section}' introuvable.")
    return m.group(1)

def _demand_grade_slice(demand_text: str, grade: str) -> str:
    other = "LSFO" if grade == "HSFO" else "HSFO"
    m_start = re.search(rf"\b{grade}\b", demand_text)
    if not m_start: raise RuntimeError(f"Grade '{grade}' introuvable dans Demand.")
    tail = demand_text[m_start.end():]
    m_end = re.search(rf"\b{other}\b", tail)
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
    ref_vals = [0]*8
    m_ref_blk = re.search(r"Ref\.?\s*Supply\s+(.*?)(?:\n\s*Blending\b|\Z)", sup, flags=re.S)
    if m_ref_blk: ref_vals = _extract_subline(m_ref_blk.group(1), grade)
    blend_vals = [0]*8
    m_bl_blk = re.search(r"\bBlending\s+(.*)$", sup, flags=re.S)
    if m_bl_blk: blend_vals = _extract_subline(m_bl_blk.group(1), grade)
    if all(v == 0 for v in ref_vals) and all(v == 0 for v in blend_vals):
        m_dir = re.search(rf"^\s*{grade}\b\s+{_tok8}", sup, flags=re.M)
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

def _parse_fig10(pdf_path: Path) -> dict:
    with pdfplumber.open(pdf_path) as pdf:
        target_page_text = None
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if "Fig 10: Europe fuel oil balance by grade" in txt:
                target_page_text = txt; break
    if not target_page_text: raise RuntimeError("Fig.10 introuvable dans le PDF.")

    patterns = {}
    for c in COUNTRIES:
        next_keywords = [k for k in COUNTRIES if k != c]
        end_pat = r"(?:" + r"|".join(map(re.escape, next_keywords + ["Source:"])) + r")"
        m = re.search(rf"{re.escape(c)}[\s\S]*?(?={end_pat})", target_page_text)
        if not m: raise RuntimeError(f"Bloc pays manquant: {c}")
        patterns[c] = m.group(0)

    data = {}
    for c, block in patterns.items():
        d = {}
        d["Balance_total"] = _extract_country_balance_agg(target_page_text, c)
        d["Demand"] = {}
        d["Supply"] = {}
        d["Demand_parts"] = {}
        d["Supply_parts"] = {}

        for grade in ["HSFO", "LSFO"]:
            inland_d, bunkers_d = _extract_demand_parts(block, grade)
            d["Demand_parts"][grade] = {"Inland": inland_d, "Bunkers": bunkers_d}
            d["Demand"][grade] = [inland_d[i] + bunkers_d[i] for i in range(8)]

            ref_s, blend_s = _extract_supply_parts(block, grade)
            d["Supply_parts"][grade] = {"Ref": ref_s, "Blend": blend_s}
            d["Supply"][grade] = [ref_s[i] + blend_s[i] for i in range(8)]

            d[f"Balance_{grade}"] = [
                (ref_s[i] + blend_s[i]) - (inland_d[i] + bunkers_d[i]) for i in range(8)
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
