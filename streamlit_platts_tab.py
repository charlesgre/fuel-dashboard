# === STREAMLIT TAB FOR PLATTS ANALYTICS ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from datetime import datetime


def generate_platts_analytics_tab():
    st.header("🧠 Platts Window Analytics")

    excel_path = "Platts window/Window platts global data.xlsx"
    df = pd.read_excel(excel_path, sheet_name="Platts window")
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'], errors='coerce')
    df['ORDER_TIME'] = pd.to_datetime(df['ORDER_TIME'], errors='coerce').dt.time
    df['DEAL_QUANTITY'] = pd.to_numeric(df['DEAL_QUANTITY'], errors='coerce')
    df.dropna(subset=['ORDER_DATE', 'BUYER', 'SELLER', 'DEAL_QUANTITY', 'HUB'], inplace=True)

    df = df[~df['HUB'].str.contains("1%", na=False)]
    grades = df['HUB'].unique()

    df['YEAR'] = df['ORDER_DATE'].dt.year
    df['MONTH_PERIOD'] = df['ORDER_DATE'].dt.to_period('M')
    df['DATE'] = df['ORDER_DATE'].dt.date
    df['DAY'] = df['ORDER_DATE'].dt.day
    df['MONTH'] = df['ORDER_DATE'].dt.strftime('%b')
    df['WEEKDAY'] = df['ORDER_DATE'].dt.day_name()
    df['HOUR'] = df['ORDER_TIME'].apply(lambda x: x.hour if pd.notnull(x) else None)
    df['BUYER'] = df['BUYER'].apply(lambda x: str(x).split()[0])
    df['SELLER'] = df['SELLER'].apply(lambda x: str(x).split()[0])

    st.success("Données chargées avec succès.")

    selected_grade = st.selectbox("🛢 Choisir un grade:", sorted(grades))
    df_grade = df[df['HUB'] == selected_grade].copy()
    current_month = pd.Timestamp.today().to_period('M')
    current_df = df_grade[df_grade['MONTH_PERIOD'] == current_month]

    if not current_df.empty:
        st.subheader(f"📊 Analyse de {selected_grade} ({current_month})")

        # Heatmap
        st.markdown("#### 🔥 Heatmap - Volumes Journaliers")
        calendar = current_df.groupby(['MONTH', 'DAY'])['DEAL_QUANTITY'].sum().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(calendar, cmap='RdBu_r', center=0, linewidths=0.5, annot=True, fmt=".1f", ax=ax)
        st.pyplot(fig)

        # Network
        st.markdown("#### 🔗 Réseau Acheteurs - Vendeurs")
        interaction_data = current_df.groupby(['BUYER', 'SELLER'])['DEAL_QUANTITY'].sum().reset_index()
        G = nx.from_pandas_edgelist(interaction_data, 'BUYER', 'SELLER', edge_attr='DEAL_QUANTITY')
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        weights = [edata['DEAL_QUANTITY'] / 10 for _, _, edata in G.edges(data=True)]
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
        nx.draw_networkx_edges(G, pos, width=weights, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        plt.axis('off')
        st.pyplot(fig)

        # Daily Volumes
        st.markdown("#### 📅 Volumes quotidiens")
        fig, ax = plt.subplots(figsize=(10, 3))
        vol = current_df.groupby('DATE')['DEAL_QUANTITY'].sum()
        vol.plot(marker='o', ax=ax)
        ax.set_title("Volumes par jour")
        ax.grid(True)
        st.pyplot(fig)

        # Buyers / Sellers
        st.markdown("#### 👥 Répartition Acheteurs / Vendeurs")
        b = current_df.groupby('BUYER')['DEAL_QUANTITY'].sum().sort_values(ascending=False)
        s = current_df.groupby('SELLER')['DEAL_QUANTITY'].sum().sort_values(ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        b.plot(kind='bar', ax=axes[0], color='skyblue')
        s.plot(kind='bar', ax=axes[1], color='orange')
        axes[0].set_title("Acheteurs")
        axes[1].set_title("Vendeurs")
        for ax in axes:
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        st.pyplot(fig)
    else:
        st.warning("Aucune donnée disponible pour le mois courant.")
