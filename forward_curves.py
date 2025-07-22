import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# === HELPERS ===
def parse_custom_date(val):
    if isinstance(val, pd.Timestamp):
        return val
    if isinstance(val, str):
        import re
        match = re.match(r"^(\d{2})[./-](\d{2})[./-](\d{4})$", val.strip())
        if match:
            day, month, year = match.groups()
            return pd.to_datetime(f"{year}-{month}-{day}", errors='coerce')
    return pd.to_datetime(val, dayfirst=True, errors='coerce')

def calculate_spreads(df, price_columns):
    df = df.sort_values(['Publish Date', 'Date'])
    for col in price_columns:
        spread_col_name = col.replace('Flat Price', 'Spread')
        df[spread_col_name] = df[col] - df.groupby('Publish Date')[col].shift(-1)
    return df

# === TULLET EXTRACTION ===
def locate_3_5_percent_barges(df):
    for index, row in df.iterrows():
        if '3.5% Barges' in row.values:
            barges_col_index = row[row == '3.5% Barges'].index[0]
            return index, barges_col_index
    return None, None

def locate_flat_prices_and_crack(df, barges_row_index):
    flat_price_row_index = barges_row_index + 1
    if flat_price_row_index >= len(df):
        return None, None, None
    flat_price_indices, crack_indices = [], []
    for col_index in range(len(df.columns)):
        val = df.iloc[flat_price_row_index, col_index]
        if val == 'Flat Price':
            flat_price_indices.append(col_index)
        elif val == 'Crack':
            crack_indices.append(col_index)
    return flat_price_row_index, flat_price_indices, crack_indices

def extract_3_5_percent_barges_prices_and_cracks_with_dates(df, publish_date):
    barges_row_index, _ = locate_3_5_percent_barges(df)
    if barges_row_index is None:
        return None
    flat_price_row_index, flat_cols, crack_cols = locate_flat_prices_and_crack(df, barges_row_index)
    if not flat_cols or not crack_cols:
        return None
    data = []
    row_index = flat_price_row_index + 1
    while row_index < len(df):
        row = df.iloc[row_index]
        if row[flat_cols + crack_cols].isnull().all():
            break
        raw_date = df.iloc[row_index, 0]
        parsed_date = parse_custom_date(raw_date)
        data.append({
            'Date': parsed_date,
            '3.5% Barges Flat Price': df.iloc[row_index, flat_cols[0]],
            '3.5% Barges Crack': df.iloc[row_index, crack_cols[0]],
            '1% Cargo Flat Price': df.iloc[row_index, flat_cols[1]],
            '1% Cargo Crack': df.iloc[row_index, crack_cols[1]],
            '0.5% Rott Marine Fuel Flat Price': df.iloc[row_index, flat_cols[2]],
            '0.5% Rott Marine Fuel Crack': df.iloc[row_index, crack_cols[2]],
            'Publish Date': publish_date
        })
        row_index += 1
    return pd.DataFrame(data)

def locate_sing_180(df):
    for index, row in df.iterrows():
        if 'SING 180' in str(row.values).upper():
            sing_180_col_index = row[row.str.upper() == 'SING 180'].index[0]
            return index, sing_180_col_index
    return None, None

def extract_sing_180_prices_and_crack_with_dates(df, publish_date):
    sing_180_row_index, sing_180_col_index = locate_sing_180(df)
    if sing_180_row_index is None:
        return None
    flat_price_row_index, flat_price_cols, crack_cols = locate_flat_prices_and_crack(df, sing_180_row_index)
    if not flat_price_cols or not crack_cols:
        return None
    data = []
    row_index = flat_price_row_index + 1
    while row_index < len(df):
        if pd.isnull(df.iloc[row_index, sing_180_col_index]):
            break
        data.append({
            'Date': df.iloc[row_index, sing_180_col_index],
            'SING 180 Flat Price': df.iloc[row_index, flat_price_cols[0]],
            'SING 380 Flat Price': df.iloc[row_index, flat_price_cols[1]],
            '0.5% SING Marine Fuel Flat Price': df.iloc[row_index, flat_price_cols[2]],
            '0.5% SING Marine Fuel Crack': df.iloc[row_index, 17],
            'Publish Date': publish_date
        })
        row_index += 1
    return pd.DataFrame(data)

def process_tullet_files(directory):
    combined_3_5 = pd.DataFrame()
    combined_sing = pd.DataFrame()
    for fname in os.listdir(directory):
        if fname.endswith(('.xls', '.xlsx')):
            try:
                date_part = fname.split(' ')[-1].split('.')[:3]
                date_str = '-'.join(date_part)
                publish_date = pd.to_datetime(date_str)
            except:
                continue
            path = os.path.join(directory, fname)
            df = pd.read_excel(path, header=None)
            barges = extract_3_5_percent_barges_prices_and_cracks_with_dates(df, publish_date)
            sing = extract_sing_180_prices_and_crack_with_dates(df, publish_date)
            if barges is not None:
                combined_3_5 = pd.concat([combined_3_5, barges], ignore_index=True)
            if sing is not None:
                combined_sing = pd.concat([combined_sing, sing], ignore_index=True)

    if combined_3_5.empty and combined_sing.empty:
        return None

    combined_3_5 = calculate_spreads(combined_3_5, ['3.5% Barges Flat Price', '1% Cargo Flat Price', '0.5% Rott Marine Fuel Flat Price'])
    combined_sing = calculate_spreads(combined_sing, ['SING 180 Flat Price', 'SING 380 Flat Price', '0.5% SING Marine Fuel Flat Price'])

    merged = pd.merge(combined_3_5, combined_sing, on=["Publish Date", "Date"], how="outer")
    merged['Euro Hi-Lo'] = merged['1% Cargo Flat Price'] - merged['3.5% Barges Flat Price']
    merged['Euro Lo5'] = merged['0.5% Rott Marine Fuel Flat Price'] - merged['1% Cargo Flat Price']
    merged['Euro High5'] = merged['0.5% Rott Marine Fuel Flat Price'] - merged['3.5% Barges Flat Price']
    merged['East Visco'] = merged['SING 180 Flat Price'] - merged['SING 380 Flat Price']
    merged['East High5'] = merged['0.5% SING Marine Fuel Flat Price'] - merged['SING 380 Flat Price']
    merged['HS E/W'] = merged['SING 380 Flat Price'] - merged['3.5% Barges Flat Price']
    merged['LS E/W'] = merged['0.5% SING Marine Fuel Flat Price'] - merged['0.5% Rott Marine Fuel Flat Price']

    return merged.drop_duplicates(subset=['Date', 'Publish Date'])

# === PLOTTING ===
def plot_forward_curve(data, index_name):
    def get_nearest_valid_date(target, all_dates, tolerance_days=3):
        candidates = [d for d in all_dates if abs((d - target).days) <= tolerance_days]
        return min(candidates, key=lambda d: abs((d - target).days)) if candidates else None

    required_columns = ['Date', 'Publish Date', index_name]
    if not all(col in data.columns for col in required_columns):
        st.warning(f"❌ Colonnes manquantes pour {index_name}.")
        return None

    df = data.dropna(subset=required_columns).copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])

    df['Month'] = df['Date'].dt.to_period('M')
    df['Publish_Month'] = df['Publish Date'].dt.to_period('M')
    df['is_m0'] = (df['Month'] == df['Publish_Month'] - 1)
    df = df.sort_values(['Publish Date', 'is_m0', 'Date'])
    df['Date_Publish_Diff'] = df.groupby('Publish Date').cumcount()

    pivot = df.pivot_table(index='Publish Date', columns='Date_Publish_Diff', values=index_name)
    pivot.columns = [f"M{i}" for i in pivot.columns]
    pivot = pivot.dropna(axis=1, how='all').sort_index()

    if pivot.empty or len(pivot) < 2:
        st.warning(f"❌ Pas assez de données pour {index_name}")
        return None

    most_recent = pivot.index[-1]
    previous = pivot.index[-2] if len(pivot) > 1 else None
    one_week = get_nearest_valid_date(most_recent - timedelta(weeks=1), pivot.index)
    one_month = get_nearest_valid_date(most_recent - timedelta(days=30), pivot.index)

    traces = []
    colors = {
        "Most Recent": "black",
        "Previous": "red",
        "One Week Ago": "green",
        "One Month Ago": "blue"
    }
    markers = {
        "Most Recent": "circle",
        "Previous": "x",
        "One Week Ago": "square",
        "One Month Ago": "triangle-up"
    }

    for label, pub_date in {
        "Most Recent": most_recent,
        "Previous": previous,
        "One Week Ago": one_week,
        "One Month Ago": one_month
    }.items():
        if pub_date not in pivot.index:
            continue
        traces.append(go.Scatter(
            x=list(pivot.columns),
            y=pivot.loc[pub_date],
            mode='lines+markers',
            name=label,
            line=dict(color=colors[label]),
            marker=dict(symbol=markers[label])
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Forward Curve – {index_name}",
        xaxis_title="Forward Month",
        yaxis_title=index_name,
        template="plotly_white",
        legend_title="Publish Date"
    )

    return fig

# === STREAMLIT TAB ===
def generate_forward_curves_tab():
    st.header("📈 Forward Curves (ARA / Singapore)")

    directory = "L:/SHARED/NEKY/Tullet"
    with st.spinner("Chargement des données Tullet..."):
        df = process_tullet_files(directory)

    if df is not None:
        st.success("Données traitées avec succès.")

        indices = [
            '3.5% Barges Flat Price', '3.5% Barges Crack', '3.5% Barges Spread',
            '1% Cargo Flat Price', '1% Cargo Spread', '0.5% Rott Marine Fuel Crack',
            '0.5% Rott Marine Fuel Spread', 'SING 180 Flat Price', 'SING 180 Spread',
            'SING 380 Flat Price', 'SING 380 Spread', '0.5% SING Marine Fuel Flat Price',
            '0.5% SING Marine Fuel Crack', '0.5% SING Marine Fuel Spread',
            'Euro Hi-Lo', 'Euro Lo5', 'Euro High5', 'East Visco', 'East High5',
            'HS E/W', 'LS E/W'
        ]

        for i in range(0, len(indices), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(indices):
                    idx = indices[i + j]
                    fig = plot_forward_curve(df, idx)
                    if fig:
                        with cols[j]:
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée valide trouvée.")