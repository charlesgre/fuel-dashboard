import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ============================ CONFIG ============================
current_date = datetime.now().strftime('%Y-%m-%d')
forecast_file = "CDD/Forecast temperatures.xlsx"
historical_file = "CDD/Data temperatures.xlsx"
output_pdf_path = f"CDD/weather_report_{current_date}.pdf"


# ======================= HISTORICAL DATA ========================
df = pd.read_excel(historical_file, skiprows=6, header=None)
df.columns = ['Date', 'Egypt_Temperature', 'Saudi_Temperature', 'Egypt_CDD', 'Saudi_CDD']
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna()
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfYear'] = df['Date'].dt.dayofyear

# ======================= FORECAST PREP ==========================
today = datetime.today()
end_date = today + timedelta(days=15)
date_range = pd.date_range(today, end_date)

forecast_days = [(d.day, d.month) for d in date_range]
df_hist = pd.read_excel(forecast_file, skiprows=6, usecols=[0, 1, 2], header=None)
df_hist.columns = ['Date', 'Egypt_Temp', 'Saudi_Temp']
df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
df_hist = df_hist.dropna(subset=['Date'])
df_hist['Day'] = df_hist['Date'].dt.day
df_hist['Month'] = df_hist['Date'].dt.month
df_hist['Year'] = df_hist['Date'].dt.year
df_filtered = df_hist[df_hist.apply(lambda row: (row['Day'], row['Month']) in forecast_days, axis=1)].copy()
df_filtered['FakeDate'] = df_filtered.apply(lambda row: datetime(2000, row['Month'], row['Day']), axis=1)

# ======================= FORECAST LOADER ========================
def load_forecast_avg(sheet_name):
    df = pd.read_excel(forecast_file, sheet_name=sheet_name, skiprows=1, usecols=[0, 1], header=None)
    df.columns = ['DateTime', 'Temperature']
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime'])
    df = df[(df['DateTime'] >= today) & (df['DateTime'] <= end_date)]
    df['Date'] = df['DateTime'].dt.date
    df_avg = df.groupby('Date')['Temperature'].mean().reset_index()
    df_avg['Date'] = pd.to_datetime(df_avg['Date'])
    df_avg['FakeDate'] = df_avg['Date'].apply(lambda d: datetime(2000, d.month, d.day))
    return df_avg

forecast_egypt = load_forecast_avg('Egypt')
forecast_saudi = load_forecast_avg('Saudi Arabia')

# ======================= GRAPH STYLE ============================
def get_line_style(year):
    if year == 2025:
        return {'color': 'black', 'linewidth': 2.5, 'label': str(year)}
    elif year == 2024:
        return {'color': 'red', 'linewidth': 2, 'label': str(year)}
    elif year == 2023:
        return {'color': 'green', 'linewidth': 2, 'label': str(year)}
    elif year == 2022:
        return {'color': '#ADD8E6', 'linewidth': 1.5, 'label': str(year)}  # Light Blue
    elif year == 2021:
        return {'color': '#FFB6C1', 'linewidth': 1.5, 'label': str(year)}  # Light Pink
    elif year == 2020:
        return {'color': '#C0C0C0', 'linewidth': 1.5, 'label': str(year)}  # Light Gray
    else:
        return {'color': 'gray', 'linewidth': 1, 'alpha': 0.3, 'label': str(year)}


# =================== HISTORICAL GRAPH (SEASONAL) ===================
import plotly.graph_objects as go
import pandas as pd

def generate_temperature_graphs_plotly(df, temp_col, name):
    base_year = 2000  # pour aligner les jours de l'année sur une même année fictive
    fig = go.Figure()

    for year in sorted(df['Year'].unique()):
        daily_temp = df[df['Year'] == year].groupby('DayOfYear')[temp_col].mean()
        dates = pd.to_datetime([f"{base_year}{str(day).zfill(3)}" for day in daily_temp.index], format='%Y%j')


        style = get_line_style(year)  # tu peux réutiliser ta fonction, adapter les couleurs plus bas
        color = style.get('color', 'gray')
        width = style.get('linewidth', 1)
        opacity = style.get('alpha', 1)

        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_temp.values,
            mode='lines',
            name=str(year),
            line=dict(color=color, width=width),
            opacity=opacity
        ))

    # Zone grisée min-max 2020-2024
    mask = df['Year'].between(2020, 2024)
    temp_range = df[mask].groupby('DayOfYear')[temp_col].agg(['min', 'max'])
    range_dates = pd.to_datetime([f"{base_year}{str(day).zfill(3)}" for day in temp_range.index], format='%Y%j')
    fig.add_trace(go.Scatter(
        x=range_dates.append(range_dates[::-1]),
        y=temp_range['min'].tolist() + temp_range['max'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(128,128,128,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='2020-2024 Range'
    ))

    fig.update_layout(
        title=f"{name} – Daily Temperatures (2020–2025)",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        xaxis=dict(tickformat="%b", dtick="M1"),
        height=400
    )
    return fig


# =================== HISTORICAL TEMP DIFFERENCE GRAPH ===================
import plotly.graph_objects as go
import calendar

def generate_temperature_graph_2_plotly(df, temp_col, name):
    comparison_years = [2020, 2021, 2022, 2023, 2024]
    months = list(range(1, 13))
    temp_2025 = df[df['Year'] == 2025].groupby('Month')[temp_col].mean().reindex(months)
    avg_temp = df[df['Year'].isin(comparison_years)].groupby('Month')[temp_col].mean().reindex(months)
    temp_2025.iloc[datetime.now().month - 1] = None  # NaN pour mois en cours

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[calendar.month_abbr[m] for m in months],
        y=avg_temp.values,
        name='2020–2024 Avg',
        marker_color='skyblue',
        width=0.4
    ))
    fig.add_trace(go.Bar(
        x=[calendar.month_abbr[m] for m in months],
        y=temp_2025.values,
        name='2025',
        marker_color='orange',
        width=0.4
    ))

    fig.update_layout(
        title=f"{name} – Monthly Temp Diff",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        barmode='group',
        height=400
    )
    return fig


# ======================= CDD GRAPH (SEASONAL) ===================
def generate_cdd_graphs_plotly(df, cdd_col, name):
    months = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in months]
    fig = go.Figure()

    for year in sorted(df['Year'].unique()):
        monthly_cdd = (
            df[(df['Year'] == year) & (df[cdd_col] > 0)]
            .groupby('Month')[cdd_col]
            .count()
            .reindex(months, fill_value=0)
        )

        if year == 2025:
            monthly_cdd[7:] = None

        style = get_line_style(year)
        color = style.get('color', 'gray')
        width = style.get('linewidth', 1)
        opacity = style.get('alpha', 1)

        fig.add_trace(go.Scatter(
            x=months,
            y=monthly_cdd.values,
            mode='lines+markers',
            name=str(year),
            line=dict(color=color, width=width),
            opacity=opacity
        ))

    # Zone grisée min-max 2020-2024
    df_range = df[(df['Year'].between(2020, 2024)) & (df[cdd_col] > 0)]
    counts = df_range.groupby(['Year', 'Month'])[cdd_col].count().unstack().reindex(index=months, fill_value=0)
    min_vals = counts.min(axis=1)
    max_vals = counts.max(axis=1)

    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=min_vals.tolist() + max_vals[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(128,128,128,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='2020–2024 Range'
    ))

    fig.update_layout(
        title=f"{name} – Monthly CDD (2020–2025)",
        xaxis=dict(tickmode='array', tickvals=months, ticktext=month_labels),
        yaxis_title="Number of CDD Days",
        height=400
    )
    return fig



# ======================= CDD DIFFERENCE GRAPH ===================
def generate_cdd_graph_2_plotly(df, cdd_col, name):
    comparison_years = [2020, 2021, 2022, 2023, 2024]
    months = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in months]

    cdd_2025 = (
        df[(df['Year'] == 2025) & (df[cdd_col] > 0)]
        .groupby('Month')[cdd_col]
        .count()
        .reindex(months, fill_value=0)
    )
    cdd_2025.iloc[7:] = None

    avg_cdd = (
        df[df['Year'].isin(comparison_years) & (df[cdd_col] > 0)]
        .groupby(['Year', 'Month'])[cdd_col]
        .count()
        .unstack(level=0)
        .mean(axis=1)
        .reindex(months, fill_value=0)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[calendar.month_abbr[m] for m in months],
        y=avg_cdd.values,
        name=f'{name} Avg 2020–2024',
        marker_color='black',
        width=0.35
    ))
    fig.add_trace(go.Bar(
        x=[calendar.month_abbr[m] for m in months],
        y=cdd_2025.values,
        name=f'{name} 2025',
        marker_color='red',
        width=0.35
    ))

    fig.update_layout(
        title=f'{name} – Monthly CDD: 2025 vs Avg',
        xaxis_title='Month',
        yaxis_title='Number of CDD Days',
        barmode='group',
        height=400
    )
    return fig


# ======================= FORECAST GRAPH ========================
def generate_forecast_graph_plotly(df_full, forecast_df, temp_col, country):
    import numpy as np
    fig = go.Figure()

    forecast_df = forecast_df.copy()
    forecast_df['FakeDate'] = forecast_df['Date'].apply(lambda d: datetime(2000, d.month, d.day))
    forecast_df = forecast_df.sort_values('FakeDate')
    forecast_days = [(d.month, d.day) for d in forecast_df['Date']]
    all_dates = forecast_df['FakeDate'].sort_values().unique()

    df_hist = df_full.copy()
    df_hist['FakeDate'] = df_hist['Date'].apply(lambda d: datetime(2000, d.month, d.day))
    df_hist['Month'] = df_hist['Date'].dt.month
    df_hist['Day'] = df_hist['Date'].dt.day
    df_hist['Year'] = df_hist['Date'].dt.year
    df_hist = df_hist[df_hist[['Month', 'Day']].apply(tuple, axis=1).isin(forecast_days)]

    df_range = df_hist[df_hist['Year'].between(2020, 2024)].copy()
    pivot = df_range.pivot_table(index='FakeDate', columns='Year', values=temp_col)
    valid_pivot = pivot.dropna(thresh=3)
    valid_dates = sorted(set(valid_pivot.index).intersection(set(all_dates)))
    min_temp = valid_pivot.loc[valid_dates].min(axis=1)
    max_temp = valid_pivot.loc[valid_dates].max(axis=1)

    fig.add_trace(go.Scatter(
        x=valid_dates + valid_dates[::-1],  # valid_dates est une liste donc on concatène simplement
        y=pd.concat([min_temp, max_temp[::-1]]).tolist(),  # pd.concat retourne une Series, on convertit en liste
        fill='toself',
        fillcolor='rgba(211,211,211,0.5)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='2020–2024 Range'
    ))



    # Tracer les années 2020–2024
    for year in sorted(df_hist['Year'].unique()):
        df_year = df_hist[df_hist['Year'] == year].sort_values('FakeDate')
        df_year = df_year[df_year['FakeDate'].isin(all_dates)]
        if not df_year.empty:
            style = get_line_style(year)
            fig.add_trace(go.Scatter(
                x=df_year['FakeDate'],
                y=df_year[temp_col],
                mode='lines',
                name=str(year),
                line=dict(color=style['color'], width=style['linewidth']),
                opacity=style.get('alpha', 1)
            ))

    # Tracer la prévision 2025
    fig.add_trace(go.Scatter(
        x=forecast_df['FakeDate'],
        y=forecast_df['Temperature'],
        mode='lines+markers',
        name='2025 Forecast',
        line=dict(color='black', dash='dash'),
        marker=dict(symbol='circle', size=6)
    ))

    fig.update_layout(
        title=f"{country} – Temperature Comparison (2020–2025)",
        xaxis_title="Date (day-month)",
        yaxis_title="Temperature (°C)",
        xaxis=dict(
            tickformat="%d-%m",
            tickmode='auto'
        ),
        height=450
    )
    return fig




# ======================= COMPARISON TABLE ======================
def create_comparison_table(df_full, forecast_df, temp_col, title, country):
    # Step 1 – Prepare forecast
    forecast_df = forecast_df.copy()
    forecast_df['Label'] = forecast_df['Date'].apply(lambda d: d.strftime('%d-%m'))
    forecast_df.set_index('Label', inplace=True)

    # Step 2 – Extract historical values for same day/month
    df_hist = df_full.copy()
    df_hist['Label'] = df_hist['Date'].apply(lambda d: d.strftime('%d-%m'))
    df_hist['Year'] = df_hist['Date'].dt.year

    # Filter for years 2020–2024 and same dates
    df_hist = df_hist[df_hist['Year'].between(2020, 2024)]
    df_hist = df_hist[df_hist['Label'].isin(forecast_df.index)]

    # Step 3 – Average historical values by date (across years)
    df_avg = df_hist.groupby('Label')[temp_col].mean()

    # Step 4 – Build the table
    df_table = pd.DataFrame({
        f"Avg {country} (2020–2024)": df_avg,
        f"Forecast {country} (2025)": forecast_df['Temperature'],
    })
    df_table["Diff (2025 - Avg)"] = df_table[f"Forecast {country} (2025)"] - df_table[f"Avg {country} (2020–2024)"]
    df_table = df_table.round(2)

    # Step 5 – Plot styled table
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(df_table)))
    ax.axis('off')

    cell_colors = []
    for _, row in df_table.iterrows():
        diff = row['Diff (2025 - Avg)']
        row_colors = ['white'] * len(row)
        if pd.notna(diff):
            if diff > 0:
                row_colors[2] = '#ff9999'
            elif diff < 0:
                row_colors[2] = '#99ccff'
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        rowLabels=df_table.index,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    plt.title(title)
    plt.tight_layout()

    return plt.gcf()  # retourne la figure active

def get_all_cdd_figures():
    figs = {}

    figs['Egypt Daily Temps'] = generate_temperature_graphs_plotly(df, 'Egypt_Temperature', 'Egypt')
    figs['Egypt Monthly Temp Diff'] = generate_temperature_graph_2_plotly(df, 'Egypt_Temperature', 'Egypt')
    figs['Egypt Monthly CDD'] = generate_cdd_graphs_plotly(df, 'Egypt_CDD', 'Egypt')
    figs['Egypt Monthly CDD Diff'] = generate_cdd_graph_2_plotly(df, 'Egypt_CDD', 'Egypt')
    figs['Egypt Temp Forecast'] = generate_forecast_graph_plotly(df, forecast_egypt, 'Egypt_Temperature', 'Egypt')
    figs['Egypt Temp Comparison Table'] = None  # Table matplotlib ignorée dans Streamlit

    figs['Saudi Daily Temps'] = generate_temperature_graphs_plotly(df, 'Saudi_Temperature', 'Saudi Arabia')
    figs['Saudi Monthly Temp Diff'] = generate_temperature_graph_2_plotly(df, 'Saudi_Temperature', 'Saudi Arabia')
    figs['Saudi Monthly CDD'] = generate_cdd_graphs_plotly(df, 'Saudi_CDD', 'Saudi Arabia')
    figs['Saudi Monthly CDD Diff'] = generate_cdd_graph_2_plotly(df, 'Saudi_CDD', 'Saudi Arabia')
    figs['Saudi Temp Forecast'] = generate_forecast_graph_plotly(df, forecast_saudi, 'Saudi_Temperature', 'Saudi Arabia')
    figs['Saudi Temp Comparison Table'] = None  # Table matplotlib ignorée dans Streamlit

    # Filtrer uniquement les figures non-nulles pour Streamlit
    figs = {k: v for k, v in figs.items() if v is not None}

    return figs

if df.empty:
    print("Attention : df est vide ! Vérifie le chargement des données.")
if forecast_egypt.empty:
    print("Attention : forecast_egypt est vide !")
if forecast_saudi.empty:
    print("Attention : forecast_saudi est vide !")