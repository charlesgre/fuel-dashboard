import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def plot_bunker_price_diffs():
    excel_path = "Bunker diff/Bunker prices excel.xlsx"
    
    df = pd.read_excel(excel_path)
    df['DATE'] = pd.to_datetime(df['ASSESSDATE'])
    df.sort_values(['SYMBOL', 'DATE'], inplace=True)
    
    quote_pairs = [
        ("PUAFI00", "PUAFZ00"), ("AAFER00", "PPXDK00"), ("AARBF00", "PUMFD00"),
        ("AARKD00", "AAPJW00"), ("AARSH00", "ICLO001"), ("AARTG00", "ICLO001"),
        ("AAXWG00", "PUAFZ00"), ("AAYJJ00", "PUABC00"), ("AUAMB00", "PUMFD00"),
        ("BFDZA00", "AAPJW00"), ("MFAGD00", "PUMFD00"), ("MFLOM00", "PUMFD00"),
        ("MFRDD00", "PUMFD00"), ("MFSPE00", "AMFSA00"), ("MFZHN00", "AMFSA00"),
        ("PUAER00", "AAPJW00"), ("PUAFA00", "PUABC00"), ("PUAFI00", "PUAFZ00"),
        ("PUAXP00", "AAIDC00"), ("PUBAD00", "PUAFZ00")
    ]
    
    required_symbols = set(s for pair in quote_pairs for s in pair)
    df_filtered = df[df['SYMBOL'].isin(required_symbols)]
    pivot = df_filtered.pivot(index='DATE', columns='SYMBOL', values='VALUE')
    
    if 'PUAFZ00' in pivot.columns:
        pivot['PUAFZ00'] = pivot['PUAFZ00'] * 6.35
    
    desc_map = df_filtered[['SYMBOL', 'DESCRIPTION']].drop_duplicates().set_index('SYMBOL')['DESCRIPTION'].to_dict()
    
    year_colors = {
        2025: 'black',
        2024: 'red',
        2023: 'green',
        2022: '#1f77b4',
        2021: 'orange',
    }
    
    cols = st.columns(3)
    col_idx = 0
    
    for sym1, sym2 in quote_pairs:
        if sym1 not in pivot.columns or sym2 not in pivot.columns:
            continue
        
        diff_series = pivot[sym1] - pivot[sym2]
        diff_df = diff_series.reset_index()
        diff_df.columns = ['DATE', 'DIFF']
        diff_df['YEAR'] = diff_df['DATE'].dt.year
        diff_df['MONTH'] = diff_df['DATE'].dt.month
        diff_df['DAY'] = diff_df['DATE'].dt.day
        diff_df['FAKE_DATE'] = pd.to_datetime(dict(year=2000, month=diff_df['MONTH'], day=diff_df['DAY']), errors='coerce')
        
        diff_df.sort_values(['YEAR', 'FAKE_DATE'], inplace=True)
        diff_df['DIFF'] = diff_df.groupby('YEAR')['DIFF'].transform(lambda g: g.interpolate())
        
        title = f"{desc_map.get(sym1, sym1)} vs {desc_map.get(sym2, sym2)}"
        
        fig = go.Figure()
        
        years_to_plot = sorted(diff_df['YEAR'].unique())
        if (sym1, sym2) in [("AARSH00", "ICLO001"), ("AARTG00", "ICLO001"), ("MFRDD00", "PUMFD00")]:
            years_to_plot = [y for y in years_to_plot if y not in [2021, 2022]]
        
        for year in years_to_plot:
            if year == 2020:
                continue
            year_data = diff_df[diff_df['YEAR'] == year]
            color = year_colors.get(year, 'grey')
            fig.add_trace(go.Scatter(
                x=year_data['FAKE_DATE'],
                y=year_data['DIFF'],
                mode='lines',
                name=str(year),
                line=dict(color=color)
            ))
        
        # Dernier point 2025
        if 2025 in diff_df['YEAR'].unique():
            last_2025 = diff_df[diff_df['YEAR'] == 2025].dropna(subset=['DIFF']).sort_values('FAKE_DATE')
            if not last_2025.empty:
                last_point = last_2025.iloc[-1]
                fig.add_annotation(
                    x=last_point['FAKE_DATE'], y=last_point['DIFF'],
                    text=f"{last_point['DIFF']:.0f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-20,
                    font=dict(size=12, color="black", family="Arial"),
                )
        
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Month",
                tickformat='%b',
                dtick="M1",
                range=[datetime(2000,1,1), datetime(2000,12,31)]
            ),
            yaxis_title="Price Difference (USD)",
            legend_title="Year",
            margin=dict(l=20, r=20, t=40, b=40),
            height=350,
            width=350,
        )
        

        cols[col_idx].plotly_chart(fig, use_container_width=True, key=f"bunker_{sym1}_{sym2}_{col_idx}")
        col_idx = (col_idx + 1) % 3
