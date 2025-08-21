import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_platts_analytics_tab():
    st.header("🧠 Platts Window Analytics (Interactif)")

    # === Chargement et prépa des données ===
    excel_path = "Platts window/Window platts global data.xlsx"
    df = pd.read_excel(excel_path, sheet_name="Platts window")
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'], errors='coerce')
    df['ORDER_TIME'] = pd.to_datetime(df['ORDER_TIME'], errors='coerce').dt.time
    df['DEAL_QUANTITY'] = pd.to_numeric(df['DEAL_QUANTITY'], errors='coerce')
    df['PRICE'] = pd.to_numeric(df.get('PRICE'), errors='coerce')
    df.dropna(subset=['ORDER_DATE', 'BUYER', 'SELLER', 'DEAL_QUANTITY', 'HUB'], inplace=True)
    df = df[~df['HUB'].str.contains("1%", na=False)]

    # Enrichissements
    df['YEAR'] = df['ORDER_DATE'].dt.year
    df['MONTH_PERIOD'] = df['ORDER_DATE'].dt.to_period('M')
    df['DATE'] = df['ORDER_DATE'].dt.date
    df['DAY'] = df['ORDER_DATE'].dt.day
    df['MONTH'] = df['ORDER_DATE'].dt.strftime('%b')
    df['HOUR'] = df['ORDER_TIME'].apply(lambda x: x.hour if pd.notnull(x) else None)
    df['BUYER'] = df['BUYER'].astype(str).str.split().str[0]
    df['SELLER'] = df['SELLER'].astype(str).str.split().str[0]

    st.success("✅ Données chargées.")

    # === Sélection du hub (grade) ===
    grades = sorted(df['HUB'].unique())
    selected_grade = st.selectbox("🛢 Choisir un hub/grade :", grades)
    df_grade = df[df['HUB'] == selected_grade]
    current_month = pd.Timestamp.today().to_period('M')
    current_df = df_grade[df_grade['MONTH_PERIOD'] == current_month]

    if current_df.empty:
        st.warning("⚠️ Aucune donnée pour le mois courant.")
        return

    st.subheader(f"📊 Analyse de {selected_grade} ({current_month})")

    # === Seasonal Diff interactif (Window - Settlement) ===
    with st.expander("📈 Seasonal Diff (Window - Settlement) — vue globale par grade", expanded=True):
        # Prix moyens "window" par date & hub
        window_prices = df.groupby(['ORDER_DATE', 'HUB'])['PRICE'].mean().reset_index()

        # Feuille "Settlement price"
        settle = pd.read_excel(excel_path, sheet_name="Settlement price", skiprows=6)
        settle = settle.rename(columns={
            settle.columns[0]: 'DATE',
            settle.columns[1]: '3.5%',
            settle.columns[2]: '0.5%',
        })
        settle['DATE'] = pd.to_datetime(settle['DATE'])

        merged = pd.merge(window_prices, settle, left_on='ORDER_DATE', right_on='DATE', how='inner')
        merged['GRADE'] = merged['HUB'].apply(lambda x: '3.5%' if '3.5' in str(x) else '0.5%')
        merged['SETTLEMENT'] = np.where(merged['GRADE'].eq('3.5%'), merged['3.5%'], merged['0.5%'])
        merged['DIFF'] = merged['PRICE'] - merged['SETTLEMENT']
        merged['Year'] = merged['DATE'].dt.year
        merged = merged[merged['Year'] >= 2023].copy()

    # Déduire automatiquement le grade (0.5% ou 3.5%) depuis le hub sélectionné
    if "0.5" in selected_grade:
        grade_choice = "0.5%"
    else:
        grade_choice = "3.5%"

    mg = merged[merged['GRADE'] == grade_choice].copy()
    if grade_choice == '0.5%':
        z = (mg['DIFF'] - mg['DIFF'].mean()) / mg['DIFF'].std(ddof=0)
        mg = mg[z.abs() < 3]


        # PseudoDate = toutes les années alignées sur 2000 pour l’effet saisonnier
        mg['PseudoDate'] = mg['DATE'].apply(lambda d: pd.Timestamp(2000, d.month, d.day))

        fig_sd = px.line(
            mg.sort_values('PseudoDate'),
            x='PseudoDate', y='DIFF', color='Year',
            labels={'PseudoDate': 'Month', 'DIFF': 'Diff (USD/tonne)'},
            title=f"Seasonal Diff (Window - Settlement) – {grade_choice}",
            markers=True
        )
        # Ticks mensuels + slider/zoom interactif
        fig_sd.update_xaxes(
            tickvals=pd.date_range("2000-01-01", "2000-12-31", freq="MS"),
            tickformat="%b",
            rangeslider_visible=True
        )
        fig_sd.update_traces(hovertemplate="%{x|%b %d} • %{fullData.name}<br>Diff: %{y:.2f}")
        st.plotly_chart(fig_sd, use_container_width=True)


    # === 1) Heatmap interactive (mois courant) ===
    st.markdown("#### 🔥 Heatmap – Volumes journaliers (mois courant)")
    heatmap_df = (
        current_df
        .groupby(['DAY', 'MONTH'])['DEAL_QUANTITY']
        .sum()
        .reset_index()
        .pivot(index='DAY', columns='MONTH', values='DEAL_QUANTITY')
        .fillna(0)
    )
    fig1 = px.imshow(
        heatmap_df,
        labels=dict(x="Mois", y="Jour", color="Volume"),
        x=heatmap_df.columns,
        y=heatmap_df.index,
        text_auto=".1f",
        aspect="auto",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # === Yearly Heatmap interactif (année en cours) ===
    st.markdown("#### 🗓️ Yearly Heatmap – Volumes journaliers (année en cours)")
    year_now = datetime.now().year
    months_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    yearly_df = df_grade[df_grade['YEAR'] == year_now].copy()
    yearly_df['MONTH'] = pd.Categorical(yearly_df['MONTH'], categories=months_order, ordered=True)

    yearly_calendar = (
        yearly_df.groupby(['MONTH', 'DAY'], observed=True)['DEAL_QUANTITY']
        .sum().unstack().reindex(index=months_order).fillna(0)
    )

    fig_y = px.imshow(
        yearly_calendar,
        labels=dict(x="Day", y="Month", color="Volume"),
        x=yearly_calendar.columns, y=yearly_calendar.index,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0
    )
    fig_y.update_layout(
        title=f"Daily Quantity Heatmap – Full Year – {selected_grade} – {year_now}",
        margin=dict(t=40, b=10, l=10, r=10),
        coloraxis_colorbar=dict(title="Volume")
    )
    # Hover lisible
    fig_y.update_traces(hovertemplate="Month: %{y}<br>Day: %{x}<br>Volume: %{z:.1f}")

    st.plotly_chart(fig_y, use_container_width=True)


    # seaborn pour colormap type report
    fig_y, ax_y = plt.subplots(figsize=(18, 6))
    sns.heatmap(yearly_calendar, cmap='RdBu_r', center=0, linewidths=0.5,
                annot=True, fmt=".1f", ax=ax_y)
    ax_y.set_title(f"Daily Quantity Heatmap – Full Year – {selected_grade} – {year_now}")
    ax_y.set_xlabel("DAY")
    ax_y.set_ylabel("MONTH")
    st.pyplot(fig_y, clear_figure=True)

    # === 3) Réseau Acheteurs–Vendeurs interactif (liens colorés) ===
    st.markdown("#### 🔗 Réseau Acheteurs – Vendeurs")
    interaction = (
        current_df
        .groupby(['BUYER', 'SELLER'])['DEAL_QUANTITY']
        .sum()
        .reset_index()
        .rename(columns={'DEAL_QUANTITY': 'QTY'})
    )
    G = nx.from_pandas_edgelist(interaction, 'BUYER', 'SELLER', edge_attr='QTY')
    pos = nx.spring_layout(G, k=0.5, seed=42)

    qmin = interaction['QTY'].min()
    qmax = interaction['QTY'].max()
    rng = (qmax - qmin) if qmax != qmin else 1.0

    def qty_to_color(q):
        t = (q - qmin) / rng
        return px.colors.sample_colorscale('Viridis', t)[0]

    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        q = d['QTY']
        edge_traces.append(
            go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(width=1 + 8*((q - qmin)/rng), color=qty_to_color(q)),
                hoverinfo='text',
                text=f"{u} → {v}<br>Quantité: {q:,.0f}",
                showlegend=False
            )
        )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text, textposition="top center",
        hoverinfo='text',
        marker=dict(size=20, color='lightblue'),
        showlegend=False
    )
    colorbar_trace = go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(colorscale='Viridis', showscale=True, cmin=qmin, cmax=qmax,
                    color=[qmin], size=0.0001,
                    colorbar=dict(title='Quantité', thickness=15, len=0.8)),
        hoverinfo='none', showlegend=False
    )
    fig2 = go.Figure(data=edge_traces + [node_trace, colorbar_trace])
    fig2.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # === 4) Volumes quotidiens ===
    st.markdown("#### 📅 Volumes quotidiens")
    daily = current_df.groupby('DATE')['DEAL_QUANTITY'].sum().reset_index()
    fig3 = px.line(
        daily, x='DATE', y='DEAL_QUANTITY',
        markers=True, labels={'DEAL_QUANTITY': 'Volume', 'DATE': 'Date'},
        title="Volumes par jour"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # === 5) Répartition Acheteurs / Vendeurs ===
    st.markdown("#### 👥 Répartition Acheteurs / Vendeurs")
    buyers = current_df.groupby('BUYER')['DEAL_QUANTITY'].sum().reset_index()
    sellers = current_df.groupby('SELLER')['DEAL_QUANTITY'].sum().reset_index()

    fig4 = px.bar(
        buyers.sort_values('DEAL_QUANTITY', ascending=False),
        x='BUYER', y='DEAL_QUANTITY',
        labels={'DEAL_QUANTITY': 'Volume', 'BUYER': 'Acheteur'},
        title="Top Acheteurs"
    )
    fig5 = px.bar(
        sellers.sort_values('DEAL_QUANTITY', ascending=False),
        x='SELLER', y='DEAL_QUANTITY',
        labels={'DEAL_QUANTITY': 'Volume', 'SELLER': 'Vendeur'},
        title="Top Vendeurs"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        st.plotly_chart(fig5, use_container_width=True)
