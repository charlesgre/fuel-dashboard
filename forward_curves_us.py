# forward_curves_us.py

import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# === CONFIG ===
quote_ranges = {
    "HS GC HS":        {"date": "B24:B32",   "value": "C24:C32"},
    "HS GC Spreads":   {"date": "E24:E32",   "value": "F24:F32"},
    "HS GC Arb":       {"date": "H24:H32",   "value": "I24:I32"},
    "GC/Brent crack":  {"date": "K24:K32",   "value": "L24:L32"},
    "GC/TI crack":     {"date": "N24:N32",   "value": "O24:O32"},
    "GC 0.5":          {"date": "Q24:Q32",   "value": "R24:R32"},
    "GC 0.5 Spread":   {"date": "T24:T32",   "value": "U24:U32"},
    "GC 0.5 Arb":      {"date": "W24:W32",   "value": "X24:X32"},
    "GC 0.5 / Brent":  {"date": "Z24:Z32",   "value": "AA24:AA32"}
}

# === HELPERS ===
def excel_range_to_index(cell):
    match = re.match(r"([A-Z]+)(\d+)", cell.upper())
    if not match:
        raise ValueError(f"Invalid Excel cell: {cell}")
    col_str, row_str = match.groups()
    col = 0
    for char in col_str:
        col = col * 26 + (ord(char) - ord('A') + 1)
    return int(row_str) - 1, col - 1

def read_us_file(file_path, publish_date):
    all_data = []
    try:
        df = pd.read_excel(file_path, header=None)
    except:
        return pd.DataFrame()

    for quote, rng in quote_ranges.items():
        try:
            d_start, d_end = rng['date'].split(':')
            v_start, v_end = rng['value'].split(':')
            d_start_r, d_col = excel_range_to_index(d_start)
            d_end_r, _ = excel_range_to_index(d_end)
            v_start_r, v_col = excel_range_to_index(v_start)
            v_end_r, _ = excel_range_to_index(v_end)
            date_vals = df.iloc[d_start_r:d_end_r+1, d_col].tolist()
            val_vals = df.iloc[v_start_r:v_end_r+1, v_col].tolist()
            if len(date_vals) != len(val_vals):
                continue
            all_data.append(pd.DataFrame({
                "Publish Date": [publish_date]*len(date_vals),
                "Quote": [quote]*len(date_vals),
                "Date": date_vals,
                "Settlement": val_vals
            }))
        except:
            continue
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# === DATA LOADING ===
def load_us_curve_data(directory):
    file_info = []
    for filename in os.listdir(directory):
        if filename.startswith("Fuel Oil Close Prices") and filename.endswith(('.xls', '.xlsx')):
            try:
                date_str = os.path.splitext(filename.split()[-1])[0]
                pub_date = datetime.strptime(date_str, "%m.%d.%Y")
                file_info.append((pub_date, os.path.join(directory, filename)))
            except:
                continue
    file_info.sort(key=lambda x: x[0])
    now = datetime.now()
    target_dates = {
        "Most Recent": now - timedelta(days=1),
        "Previous": now - timedelta(days=2),
        "One Week Before": now - timedelta(days=7),
        "One Month Before": now - timedelta(days=30)
    }
    used_files = set()
    selected = {}
    for label, tdate in target_dates.items():
        candidates = [(d, f) for d, f in file_info if f not in used_files]
        if not candidates:
            continue
        if label == "Most Recent":
            best = max(candidates, key=lambda x: x[0])
        elif label == "Previous" and "Most Recent" in selected:
            mr = selected["Most Recent"][0]
            past = [x for x in candidates if x[0] < mr]
            best = max(past, key=lambda x: x[0]) if past else max(candidates, key=lambda x: x[0])
        else:
            best = min(candidates, key=lambda x: abs((x[0] - tdate).total_seconds()))
        selected[label] = best
        used_files.add(best[1])

    frames = []
    for label, (pub_date, path) in selected.items():
        df = read_us_file(path, pub_date)
        if not df.empty:
            df['Target'] = label
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()

# === PLOTTING ===
def plot_us_forward_curve(df, quote):
    df = df[df['Quote'] == quote].dropna(subset=['Settlement'])
    if df.empty:
        return None
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    df['MIndex'] = df.groupby('Publish Date').cumcount().map(lambda x: f"M{x}")
    pivot = df.pivot_table(index='Publish Date', columns='MIndex', values='Settlement')
    if pivot.empty:
        return None
    x = list(range(len(pivot.columns)))
    x_labels = list(pivot.columns)
    fig = go.Figure()
    colors = {
        "Most Recent": "black",
        "Previous": "red",
        "One Week Before": "green",
        "One Month Before": "blue"
    }
    for pub_date in pivot.index:
        label = df[df['Publish Date'] == pub_date]['Target'].iloc[0]
        y = pivot.loc[pub_date].ffill().bfill()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=label,
            marker=dict(symbol='circle'),
            line=dict(color=colors.get(label, 'gray'))
        ))
    fig.update_layout(
        title=f"US Forward Curve – {quote}",
        xaxis=dict(tickvals=x, ticktext=x_labels, title="Forward Month"),
        yaxis=dict(title=quote),
        template="plotly_white",
        legend=dict(title="Publish Date")
    )
    return fig

# === STREAMLIT TAB ===
def generate_us_forward_curves_tab():
    st.header("📈 US Forward Curves")

    directory = "Forward curves US/US closure"
    with st.spinner("Chargement des données..."):
        df = load_us_curve_data(directory)

    if df is not None and not df.empty:
        st.success("Données chargées avec succès.")
        quotes = df['Quote'].unique().tolist()

        for i in range(0, len(quotes), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(quotes):
                    quote = quotes[i + j]
                    fig = plot_us_forward_curve(df, quote)
                    if fig:
                        with cols[j]:
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée disponible.")
