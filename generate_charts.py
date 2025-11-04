import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ⚠️ Chemin vers ton fichier Excel
FILE_PATH = "Prices/Prices sheet.xlsx"

# 🎯 Titres ciblés (VGO supprimé)
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

# Couleurs par année
year_colors = {
    2022: "gray",
    2023: "gold",
    2024: "green",
    2025: "red",
}

# —— Utilitaire pour éviter que Plotly/MathJax interprète le signe $ comme du LaTeX
def safe_plotly_text(text: str) -> str:
    """Échappe les $ pour empêcher MathJax de passer en mode 'math'."""
    if text is None:
        return ""
    return text.replace("$", r"\$")

# —— Chargement des données
def load_excel_data():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Fichier Excel introuvable: {FILE_PATH}")
    try:
        xl = pd.ExcelFile(FILE_PATH)
        df_data = xl.parse("Data", header=None)
        return df_data
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du fichier Excel: {e}")

# —— Préparation des données
def prepare_data(df_raw, titles_row=3, start_row=7):
    date_col = 0
    dates = pd.to_datetime(df_raw.iloc[start_row:, date_col], errors="coerce")
    data = {}
    for col in range(1, df_raw.shape[1]):
        title = df_raw.iloc[titles_row, col]
        values = pd.to_numeric(df_raw.iloc[start_row:, col], errors="coerce")
        temp_df = pd.DataFrame({"Date": dates, "Value": values}).dropna()
        if temp_df.empty:
            continue
        temp_df["Year"] = temp_df["Date"].dt.year
        temp_df["DayOfYear"] = temp_df["Date"].dt.dayofyear
        data[title] = temp_df
    return data

# —— Filtre des outliers simple (z-score)
def remove_outliers(df, column="Value", threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[z_scores.abs() < threshold]

# —— Graphe interactif saisonnalité (par année)
def generate_interactive_chart(df, title):
    fig = go.Figure()

    for year in sorted(df["Year"].unique()):
        if year not in year_colors:
            continue
        year_data = df[df["Year"] == year]
        year_data = remove_outliers(year_data, "Value")
        if year_data.empty:
            continue

        # Ramène toutes les dates sur une année "référence" pour la saisonnalité
        ref_dates = pd.to_datetime("2000-01-01") + pd.to_timedelta(
            year_data["DayOfYear"] - 1, unit="D"
        )

        fig.add_trace(go.Scatter(
            x=ref_dates,
            y=year_data["Value"],
            mode="lines",
            name=str(year),
            line=dict(color=year_colors[year]),
        ))

    fig.update_layout(
        title=safe_plotly_text(f"Seasonality - {title}"),
        xaxis_title="Month",
        yaxis_title="Value",
        xaxis=dict(tickformat="%b", dtick="M1"),
        template="plotly_white",
        height=500,
    )
    return fig

# —— Génère les graphiques pour une sélection de titres
def generate_price_charts(selected_titles=None):
    df_raw = load_excel_data()
    all_data = prepare_data(df_raw)

    if selected_titles is None:
        selected_titles = target_titles

    charts = {}
    for title in selected_titles:
        if title in all_data and not all_data[title].empty:
            charts[title] = generate_interactive_chart(all_data[title], title)
    return charts

# —— Diffs NWE vs MED
def generate_nwe_med_diff_charts():
    """
    Lit l’onglet 'NWE -MED diffs', calcule (V1 - V2) pour chaque paire de SYMBOL,
    et renvoie un dict { 'Diff <DESC1> vs <DESC2>': figure, ... }.
    (Les codes type PUABC00 ne sont pas affichés.)
    """
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Fichier Excel introuvable: {FILE_PATH}")

    xl = pd.ExcelFile(FILE_PATH)
    # attend : colonnes SYMBOL, DESCRIPTION, ASSESSDATE, VALUE
    df_diffs = xl.parse("NWE -MED diffs")

    # Nettoyage des types/valeurs
    df_diffs["ASSESSDATE"] = pd.to_datetime(df_diffs.get("ASSESSDATE"), errors="coerce")
    df_diffs["VALUE"] = pd.to_numeric(df_diffs.get("VALUE"), errors="coerce")
    df_diffs = df_diffs.dropna(subset=["ASSESSDATE", "VALUE", "SYMBOL"])

    # Dictionnaire SYMBOL -> DESCRIPTION (si dispo)
    desc_map = {}
    if "DESCRIPTION" in df_diffs.columns:
        desc_map = (
            df_diffs.drop_duplicates("SYMBOL")
                    .set_index("SYMBOL")["DESCRIPTION"]
                    .to_dict()
        )

    # Paires identiques à ton script matplotlib
    diff_pairs = [
        ("PUABC00", "PUAAZ00"),
        ("PUABA00", "PUAAY00"),
        ("PUAAM00", "PUAAK00"),
        ("PUAAL00", "PUAAJ00"),
        ("PUMFD00", "MFFMM00"),
    ]

    charts = {}

    for sym1, sym2 in diff_pairs:
        sub1 = (df_diffs[df_diffs["SYMBOL"] == sym1][["ASSESSDATE", "VALUE"]]
                .rename(columns={"VALUE": "V1"}))
        sub2 = (df_diffs[df_diffs["SYMBOL"] == sym2][["ASSESSDATE", "VALUE"]]
                .rename(columns={"VALUE": "V2"}))

        merged = pd.merge(sub1, sub2, on="ASSESSDATE", how="inner").dropna()
        if merged.empty:
            continue

        # Calcul du diff et colonnes temporelles
        merged["diff"] = merged["V1"] - merged["V2"]
        merged["Year"] = merged["ASSESSDATE"].dt.year
        merged["DayOfYear"] = merged["ASSESSDATE"].dt.dayofyear
        merged = merged[merged["Year"].isin(year_colors.keys())]
        if merged.empty:
            continue

        # Figure de saisonnalité
        fig = go.Figure()
        for year in sorted(merged["Year"].unique()):
            if year not in year_colors:
                continue
            yd = merged[merged["Year"] == year].copy()
            if yd.empty:
                continue
            ref_dates = pd.to_datetime("2000-01-01") + pd.to_timedelta(
                yd["DayOfYear"] - 1, unit="D"
            )
            fig.add_trace(go.Scatter(
                x=ref_dates,
                y=yd["diff"],
                mode="lines",
                name=str(year),
                line=dict(color=year_colors[year]),
            ))

        # Titre lisible (sans codes) et SAFE pour Plotly
        title_txt = f"Diff {desc_map.get(sym1, sym1)} vs {desc_map.get(sym2, sym2)}"
        fig.update_layout(
            title=safe_plotly_text(title_txt),
            xaxis_title="Month",
            yaxis_title="Diff (USD/MT)",
            xaxis=dict(tickformat="%b", dtick="M1"),
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", y=-0.2),
        )

        # ✅ Clé d’affichage = titre lisible
        charts[title_txt] = fig

    return charts
