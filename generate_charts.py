import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ⚠️ Chemin corrigé vers ton fichier Excel
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

year_colors = {
    2022: 'gray',
    2023: 'gold',
    2024: 'green',
    2025: 'red'
}

def load_excel_data():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Fichier Excel introuvable: {FILE_PATH}")
    try:
        xl = pd.ExcelFile(FILE_PATH)
        df_data = xl.parse("Data", header=None)
        return df_data
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du fichier Excel: {e}")

def prepare_data(df_raw, titles_row=3, start_row=7):
    date_col = 0
    dates = pd.to_datetime(df_raw.iloc[start_row:, date_col], errors='coerce')
    data = {}
    for col in range(1, df_raw.shape[1]):
        title = df_raw.iloc[titles_row, col]
        values = pd.to_numeric(df_raw.iloc[start_row:, col], errors='coerce')
        temp_df = pd.DataFrame({"Date": dates, "Value": values}).dropna()
        temp_df["Year"] = temp_df["Date"].dt.year
        temp_df["DayOfYear"] = temp_df["Date"].dt.dayofyear
        data[title] = temp_df
    return data

def remove_outliers(df, column="Value", threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[z_scores.abs() < threshold]

def generate_interactive_chart(df, title):
    fig = go.Figure()
    for year in sorted(df['Year'].unique()):
        if year in year_colors:
            year_data = df[df['Year'] == year]
            year_data = remove_outliers(year_data, "Value")
            if year_data.empty:
                continue
            ref_dates = pd.to_datetime('2000-01-01') + pd.to_timedelta(year_data['DayOfYear'] - 1, unit='D')
            fig.add_trace(go.Scatter(
                x=ref_dates,
                y=year_data['Value'],
                mode='lines',
                name=str(year),
                line=dict(color=year_colors[year])
            ))
    fig.update_layout(
        title=f"Seasonality - {title}",
        xaxis_title="Month",
        yaxis_title="Value",
        xaxis=dict(tickformat="%b", dtick="M1"),
        template="plotly_white",
        height=500
    )
    return fig

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
