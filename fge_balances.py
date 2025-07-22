import os
import glob
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Chemin vers le dossier contenant les fichiers Excel
folder_path = r"\\gvaps1\USR6\CHGE\desktop\Fuel desk\EA vs FGE balance\FGE balances"

def get_latest_excel_file():
    excel_files = glob.glob(os.path.join(folder_path, "*.xls*"))
    if not excel_files:
        raise FileNotFoundError(f"Aucun fichier Excel trouvé dans {folder_path}")
    latest_file = max(excel_files, key=os.path.getmtime)
    return latest_file

def extract_clean_balance(start_index, df, targets):
    # Extraction et nettoyage de la section balance du dataframe
    dates = df.iloc[start_index + 1, 1:].values
    section_df = df.iloc[start_index + 2:, :].copy()
    section_df.columns = ["Region"] + list(dates)
    section_df = section_df.drop_duplicates(subset=["Region"])
    section_df = section_df[section_df["Region"].isin(targets)]
    section_df.set_index("Region", inplace=True)
    section_df.columns = pd.to_datetime(section_df.columns)
    return section_df.T  # Transpose pour avoir dates en index

def load_fge_balances():
    latest_file = get_latest_excel_file()
    excel_file = pd.ExcelFile(latest_file)
    balance_df = excel_file.parse("Balance", header=None)
    vlsfo_start = balance_df[balance_df[0] == "VLSFO Balance (kb/d)"].index[0]
    hsfo_start = balance_df[balance_df[0] == "HSFO Balance (kb/d)"].index[0]

    regions = ["Europe", "Belgium", "Netherlands", "Egypt", "North West", "Mediterranean", "Saudi Arabia"]
    vlsfo_data = extract_clean_balance(vlsfo_start, balance_df, regions)
    hsfo_data = extract_clean_balance(hsfo_start, balance_df, regions)

    return vlsfo_data, hsfo_data

def plot_fge_balances(data, fuel_type):
    figs = {}
    for region in data.columns:
        df = data[[region]].copy()
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        pivot = df.pivot_table(index='Month', columns='Year', values=region)

        fig = go.Figure()
        if 2025 in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[2025], mode='lines+markers', name='2025', line=dict(color='black')))
        if 2026 in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[2026], mode='lines+markers', name='2026', line=dict(color='red')))

        fig.update_layout(
            title=f"{fuel_type} - {region}",
            xaxis=dict(
                title='Month',
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ),
            yaxis_title='kb/d',
            height=400,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        figs[f"{fuel_type} - {region}"] = fig
    return figs
