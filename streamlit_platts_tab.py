import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

def generate_platts_analytics_tab():
    st.header("🧠 Platts Window Analytics (Interactif)")

    # Chargement et prépa des données
    excel_path = "Platts window/Window platts global data.xlsx"
    df = pd.read_excel(excel_path, sheet_name="Platts window")
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'], errors='coerce')
    df['ORDER_TIME'] = pd.to_datetime(df['ORDER_TIME'], errors='coerce').dt.time
    df['DEAL_QUANTITY'] = pd.to_numeric(df['DEAL_QUANTITY'], errors='coerce')
    df.dropna(subset=['ORDER_DATE', 'BUYER', 'SELLER', 'DEAL_QUANTITY', 'HUB'], inplace=True)
    df = df[~df['HUB'].str.contains("1%", na=False)]

    # Enrichissements temporels
    df['MONTH_PERIOD'] = df['ORDER_DATE'].dt.to_period('M')
    df['DATE'] = df['ORDER_DATE'].dt.date
    df['DAY'] = df['ORDER_DATE'].dt.day
    df['MONTH'] = df['ORDER_DATE'].dt.strftime('%b')
    df['HOUR'] = df['ORDER_TIME'].apply(lambda x: x.hour if pd.notnull(x) else None)
    df['BUYER'] = df['BUYER'].astype(str).str.split().str[0]
    df['SELLER'] = df['SELLER'].astype(str).str.split().str[0]

    st.success("✅ Données chargées.")

    # Sélection du grade
    grades = sorted(df['HUB'].unique())
    selected_grade = st.selectbox("🛢 Choisir un grade :", grades)
    df_grade = df[df['HUB'] == selected_grade]
    current_month = pd.Timestamp.today().to_period('M')
    current_df = df_grade[df_grade['MONTH_PERIOD'] == current_month]

    if current_df.empty:
        st.warning("⚠️ Aucune donnée pour le mois courant.")
        return

    st.subheader(f"📊 Analyse de {selected_grade} ({current_month})")

    # 1) Heatmap interactive
    st.markdown("#### 🔥 Heatmap – Volumes journaliers")
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

    # 2) Réseau Acheteurs–Vendeurs interactif
    st.markdown("#### 🔗 Réseau Acheteurs – Vendeurs")
    interaction = (
        current_df
        .groupby(['BUYER', 'SELLER'])['DEAL_QUANTITY']
        .sum()
        .reset_index()
    )
    G = nx.from_pandas_edgelist(interaction, 'BUYER', 'SELLER', edge_attr='DEAL_QUANTITY')
    pos = nx.spring_layout(G, k=0.5, seed=42)

    edge_x, edge_y, edge_width = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_width.append(d['DEAL_QUANTITY'] / interaction['DEAL_QUANTITY'].max() * 10)

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(size=20, color='lightblue')
    )

    fig2 = go.Figure(data=[edge_trace, node_trace])
    fig2.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False),
                       margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Volumes quotidiens interactif
    st.markdown("#### 📅 Volumes quotidiens")
    daily = current_df.groupby('DATE')['DEAL_QUANTITY'].sum().reset_index()
    fig3 = px.line(
        daily, x='DATE', y='DEAL_QUANTITY',
        markers=True, labels={'DEAL_QUANTITY': 'Volume', 'DATE': 'Date'},
        title="Volumes par jour"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 4) Répartition Acheteurs / Vendeurs interactif
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
