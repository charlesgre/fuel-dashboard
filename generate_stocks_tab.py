import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np

FILE_PATH = "Fuel dashboard/Stocks/Fuel stocks.xlsx"

sheets_config = {
    'Data ARA PJK': {'start_row': 6, 'date_col': 'A', 'data_col': 'B'},
    'Data PADD 3': {'start_row': 7, 'date_col': 'A', 'data_col': 'B'},
    'Data Singapore': {'start_row': 7, 'date_col': 'A', 'data_col': 'B'},
    'Fujairah datas': {'start_row': 3, 'date_col': 'A', 'data_col': 'G'}
}

colors = {
    2020: 'blue',
    2021: 'orange',
    2022: 'purple',
    2023: 'yellow',
    2024: 'green',
    2025: 'red'
}

def load_and_prepare(sheet_name, config):
    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name, header=None, skiprows=config['start_row'])
    dates = pd.to_datetime(df[ord(config['date_col']) - 65], errors='coerce')
    values = pd.to_numeric(df[ord(config['data_col']) - 65], errors='coerce')
    df_clean = pd.DataFrame({'Date': dates, 'Value': values}).dropna()

    if sheet_name.strip() == "Data ARA PJK":
        df_clean['Value'] *= 6.35 / 1000
    elif sheet_name.strip() in ["Data PADD 3", "Data Singapore"]:
        df_clean['Value'] /= 1000

    df_clean['Year'] = df_clean['Date'].dt.year
    df_clean['DOY'] = df_clean['Date'].dt.dayofyear
    return df_clean

def build_comparison_table(df):
    latest_date = df['Date'].max()
    latest_value = df[df['Date'] == latest_date]['Value'].values[0]

    compare_dates = {
        'Previous week': latest_date - timedelta(days=7),
        'Previous month': latest_date - timedelta(days=30),
        'Previous year': latest_date - timedelta(days=365),
    }

    records = [{
        'Label': 'Latest',
        'Date': latest_date.strftime('%d-%m-%Y'),
        'Value (MMB)': round(latest_value, 2),
        'Change (MMB)': '-',
        'Change %': '-'
    }]

    for label, target_date in compare_dates.items():
        closest = df.iloc[(df['Date'] - target_date).abs().argsort()[:1]]
        comp_date = closest['Date'].values[0]
        comp_value = closest['Value'].values[0]
        delta_net = latest_value - comp_value
        delta_pct = (delta_net / comp_value * 100) if comp_value != 0 else 0

        records.append({
            'Label': label,
            'Date': pd.to_datetime(comp_date).strftime('%d-%m-%Y'),
            'Value (MMB)': round(comp_value, 2),
            'Change (MMB)': round(delta_net, 2),
            'Change %': f"{round(delta_pct, 2)}%"
        })

    return pd.DataFrame(records)

def generate_stocks_tab():
    st.write("✅ Tab Stocks appelée")
    st.header("📦 Seasonal Stocks Overview")

    for sheet_name, config in sheets_config.items():
        st.markdown(f"## 📁 {sheet_name}")

        df = load_and_prepare(sheet_name, config)
        fig = go.Figure()

        for year in sorted(df['Year'].unique()):
            data_year = df[df['Year'] == year]
            fig.add_trace(go.Scatter(
                x=data_year['DOY'],
                y=data_year['Value'],
                mode='lines',
                name=str(year),
                line=dict(color=colors.get(year, 'gray')),
                opacity=1.0 if year >= 2023 else 0.6
            ))

        fig.update_layout(
            title=f"{sheet_name} – Évolution saisonnière",
            xaxis_title="Jour de l'année",
            yaxis_title="Stock (MMB)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tableau comparatif
        st.markdown("#### 🔍 Comparatif temporel")
        compare_df = build_comparison_table(df)
        st.dataframe(compare_df, use_container_width=True)
        st.markdown("---")
