import os
from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go

# ========= Config =========
FILE_PATH = "Prices/Prices sheet.xlsx"

year_colors = {
    2022: "gray",
    2023: "gold",
    2024: "green",
    2025: "red",
}

# Titres ciblés (benchmarks)
target_titles = [
    "EUR FO 3.5 FOB Rdam Swap", "Brent Frontline", "Rotterdam Gasoil 0.1%",
    "180Middle east vs 180Singap", "No6 3.0 Gulf", "Barges spot crack",
    "Barges Crack spot ratio", "Visco", "Hilo", "M1/M2 380 CST spread",
    "M1/M2 Barges spread", "M1/M2 0.5 Rotter spread", "M1/M2 0.5 Singap spread",
    "HSFO E/W M1spread", "0.5 Rotter M1", "High 5 Rotterdam", "1% FO Rotterdam",
    "Lo5", "FOGO", "0.5% East/West", "TD20 M1", "380 cracks M1",
    "3.5B M0/M1", "380 cracks vs Arab Medium", "0.5 cracks vs WTI landed",
    "0.5 Rott cracks M1", "0.5 Singap cracks"
]

# ========= Utils =========
def safe_plotly_text(text: str) -> str:
    """Empêche MathJax de traiter les $ comme du LaTeX dans les titres."""
    return (text or "").replace("$", r"\$")

def remove_outliers(df, column="Value", threshold=3):
    z = (df[column] - df[column].mean()) / df[column].std()
    return df[z.abs() < threshold]

def _load_excel() -> pd.ExcelFile:
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Fichier Excel introuvable: {FILE_PATH}")
    return pd.ExcelFile(FILE_PATH)

# ========= Benchmarks (onglet Data) =========
def _load_data_sheet():
    xl = _load_excel()
    return xl.parse("Data", header=None)

def _prepare_data(df_raw, titles_row=3, start_row=7):
    date_col = 0
    dates = pd.to_datetime(df_raw.iloc[start_row:, date_col], errors="coerce")
    data = {}
    for col in range(1, df_raw.shape[1]):
        title = df_raw.iloc[titles_row, col]
        values = pd.to_numeric(df_raw.iloc[start_row:, col], errors="coerce")
        tmp = pd.DataFrame({"Date": dates, "Value": values}).dropna()
        if tmp.empty:
            continue
        tmp["Year"] = tmp["Date"].dt.year
        tmp["DayOfYear"] = tmp["Date"].dt.dayofyear
        data[title] = tmp
    return data

def _seasonality_chart(df, title, ytitle="Value"):
    fig = go.Figure()
    for year in sorted(df["Year"].unique()):
        if year not in year_colors:
            continue
        yd = remove_outliers(df[df["Year"] == year], "Value")
        if yd.empty:
            continue
        ref_dates = pd.to_datetime("2000-01-01") + pd.to_timedelta(yd["DayOfYear"] - 1, unit="D")
        fig.add_trace(go.Scatter(
            x=ref_dates, y=yd["Value"], mode="lines",
            name=str(year), line=dict(color=year_colors[year])
        ))
    fig.update_layout(
        title=safe_plotly_text(title),
        xaxis_title="Month", yaxis_title=ytitle,
        xaxis=dict(tickformat="%b", dtick="M1"),
        template="plotly_white", height=500
    )
    return fig

def build_benchmark_section():
    df_raw = _load_data_sheet()
    all_series = _prepare_data(df_raw)
    charts = {}
    for title in target_titles:
        if title in all_series and not all_series[title].empty:
            charts[title] = _seasonality_chart(all_series[title], f"Seasonality - {title}")
    return charts

# ========= NWE – MED diffs =========
def build_nwe_med_diffs_section():
    xl = _load_excel()
    df = xl.parse("NWE -MED diffs")

    df["ASSESSDATE"] = pd.to_datetime(df.get("ASSESSDATE"), errors="coerce")
    df["VALUE"] = pd.to_numeric(df.get("VALUE"), errors="coerce")
    df = df.dropna(subset=["ASSESSDATE", "VALUE", "SYMBOL"])

    desc_map = {}
    if "DESCRIPTION" in df.columns:
        desc_map = (df.drop_duplicates("SYMBOL")
                     .set_index("SYMBOL")["DESCRIPTION"].to_dict())

    diff_pairs = [
        ("PUABC00", "PUAAZ00"),
        ("PUABA00", "PUAAY00"),
        ("PUAAM00", "PUAAK00"),
        ("PUAAL00", "PUAAJ00"),
        ("PUMFD00", "MFFMM00"),
    ]

    charts = {}
    for s1, s2 in diff_pairs:
        a = (df[df["SYMBOL"] == s1][["ASSESSDATE", "VALUE"]]
             .rename(columns={"VALUE": "V1"}))
        b = (df[df["SYMBOL"] == s2][["ASSESSDATE", "VALUE"]]
             .rename(columns={"VALUE": "V2"}))
        merged = pd.merge(a, b, on="ASSESSDATE", how="inner").dropna()
        if merged.empty:
            continue

        merged["Value"] = merged["V1"] - merged["V2"]
        merged["Year"] = merged["ASSESSDATE"].dt.year
        merged["DayOfYear"] = merged["ASSESSDATE"].dt.dayofyear
        merged = merged[merged["Year"].isin(year_colors.keys())]
        if merged.empty:
            continue

        nice_title = f"Diff {desc_map.get(s1, s1)} vs {desc_map.get(s2, s2)}"
        fig = _seasonality_chart(merged[["Value", "Year", "DayOfYear"]].copy(),
                                 nice_title,
                                 ytitle="Diff (USD/MT)")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        charts[nice_title] = fig
    return charts

# ========= VGO (Spread vs Brent) =========
def _latest_brent_from_data_sheet(start_row=7, brent_col_idx=2, date_col=0) -> float:
    df = _load_data_sheet()
    brent = pd.to_numeric(df.iloc[start_row:, brent_col_idx], errors="coerce")
    dates = pd.to_datetime(df.iloc[start_row:, date_col], errors="coerce")
    brent_df = pd.DataFrame({"Date": dates, "Brent": brent}).dropna()
    if brent_df.empty:
        raise ValueError("Impossible de récupérer le Brent depuis 'Data'.")
    return brent_df.sort_values("Date")["Brent"].iloc[-1]

def _usd_per_bbl(row):
    if row["UOM"] == "BBL" and row["CURRENCY"] == "USD":
        return row["VALUE"]
    elif row["UOM"] == "MT" and row["CURRENCY"] == "USD":
        return row["VALUE"] / 7.33
    else:
        return None

def build_vgo_section():
    xl = _load_excel()
    vgo = xl.parse("VGO prices")
    vgo = vgo[["DESCRIPTION", "ASSESSDATE", "VALUE", "UOM", "CURRENCY"]].dropna()

    vgo["ASSESSDATE"] = pd.to_datetime(vgo["ASSESSDATE"], errors="coerce")
    vgo["VALUE"] = pd.to_numeric(vgo["VALUE"], errors="coerce")
    vgo = vgo.dropna(subset=["ASSESSDATE", "VALUE"])

    vgo["USD/BBL"] = vgo.apply(_usd_per_bbl, axis=1)
    vgo = vgo.dropna(subset=["USD/BBL"])

    latest_brent = _latest_brent_from_data_sheet()
    vgo["Value"] = vgo["USD/BBL"] - latest_brent       # Spread vs Brent
    vgo["Year"] = vgo["ASSESSDATE"].dt.year
    vgo["DayOfYear"] = vgo["ASSESSDATE"].dt.dayofyear
    vgo = vgo[vgo["Year"].isin(year_colors)]

    charts = {}
    for vgo_type in sorted(vgo["DESCRIPTION"].dropna().unique()):
        sub = vgo[vgo["DESCRIPTION"] == vgo_type]
        if sub.empty:
            continue
        # On réutilise la même fonction de graphe (colonnes identiques : Value/Year/DayOfYear)
        title = f"Seasonality – {vgo_type} Spread vs Brent"
        fig = _seasonality_chart(sub[["Value", "Year", "DayOfYear"]].copy(),
                                 title,
                                 ytitle="USD/bbl (VGO – Brent)")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        charts[title] = fig
    return charts

# ========= Assemblage global =========
def generate_all_prices():
    """
    Retourne un OrderedDict avec 3 sections :
    - 'Benchmarks' : { titre: figure }
    - 'NWE - MED diffs' : { titre: figure }
    - 'VGO' : { titre: figure }
    """
    sections = OrderedDict()
    sections["Benchmarks"] = build_benchmark_section()
    sections["NWE - MED diffs"] = build_nwe_med_diffs_section()
    sections["VGO"] = build_vgo_section()
    return sections

# ===== Exemple d’usage =====
# sections = generate_all_prices()
# sections["Benchmarks"]        -> dict de figures
# sections["NWE - MED diffs"]   -> dict de figures
# sections["VGO"]               -> dict de figures
