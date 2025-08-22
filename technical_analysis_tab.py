# technical_analysis_tab.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_FILE = "Prices/Prices sheet.xlsx"  # adapte si besoin

SECURITIES = [
    "EUR FO 3.5 FOB Rdam Swap", "Barges spot crack", "3.5 Barges M1",
    "380 CST M1", "M1/M2 380 CST spread", "M1/M2 Barges spread",
    "0.5 Rotter M1", "M1/M2 0.5 Rotter spread", "0.5 Singap M1",
    "M1/M2 0.5 Singap spread", "HSFO E/W M1spread", "High 5 Rotterdam"
]

def _compute_indicators(series: pd.Series, bollinger_period: int = 20):
    # filtre outliers (|z|>3)
    z = (series - series.mean()) / series.std()
    series = series[z.abs() <= 3]
    # lissage
    series = series.rolling(3, center=True, min_periods=1).mean().dropna()

    sma = series.rolling(bollinger_period).mean()
    std = series.rolling(bollinger_period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).clip(upper=100)

    mean_dev = series.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = ((series - sma) / (0.015 * mean_dev)).clip(-400, 400)
    z_score = ((series - sma) / std).clip(-4, 4)

    return pd.DataFrame({
        "price": series,
        "sma": sma,
        "upper": upper,
        "lower": lower,
        "rsi": rsi,
        "cci": cci,
        "z": z_score
    }).dropna()

def _analyze_signals(row):
    sig = {}
    # Bollinger
    if row["price"] > row["upper"]:
        sig["Bollinger"] = "Sell"
    elif row["price"] < row["lower"]:
        sig["Bollinger"] = "Buy"
    else:
        sig["Bollinger"] = "Neutral"
    # RSI
    if row["rsi"] > 70:
        sig["RSI"] = "Sell"
    elif row["rsi"] < 30:
        sig["RSI"] = "Buy"
    else:
        sig["RSI"] = "Neutral"
    # CCI
    if row["cci"] > 100:
        sig["CCI"] = "Sell"
    elif row["cci"] < -100:
        sig["CCI"] = "Buy"
    else:
        sig["CCI"] = "Neutral"
    # Z-score
    if row["z"] > 1:
        sig["Z-Score"] = "Sell"
    elif row["z"] < -1:
        sig["Z-Score"] = "Buy"
    else:
        sig["Z-Score"] = "Neutral"
    return sig

def _signal_color(text):
    return {"Buy": "green", "Sell": "red", "Neutral": "gray"}.get(text, "gray")

def _load_prices(uploaded_file):
    # 1) Résolution du chemin par défaut (robuste)
    if uploaded_file is None:
        from pathlib import Path
        base = Path(__file__).resolve().parent
        xlsx_path = base / "Prices" / "Prices sheet.xlsx"
        if not xlsx_path.exists():
            st.error(f"Fichier introuvable: {xlsx_path}")
            return pd.DataFrame()
        excel_file = pd.ExcelFile(xlsx_path, engine="openpyxl")
    else:
        excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")

    # 2) Recherche automatique de la bonne feuille + ligne d'entêtes
    best_df = None
    best_score = -1
    best_info = None

    for sheet in excel_file.sheet_names:
        # on lit d'abord sans entêtes pour scanner les premières ~30 lignes
        tmp = pd.read_excel(excel_file, sheet_name=sheet, header=None, nrows=50)
        tmp = tmp.fillna("")

        header_row = None
        score_row = -1

        for i in range(min(30, len(tmp))):
            row_vals = [str(x).strip() for x in list(tmp.iloc[i, :].values)]
            row_lower = [v.lower() for v in row_vals]

            # heuristique 1 : présence d'une colonne Date
            has_date = any(v in ("date", "dates") for v in row_lower)

            # heuristique 2 : nb de correspondances avec nos SECURITIES
            matches = 0
            for sec in SECURITIES:
                if any(sec.lower() == v for v in row_lower):
                    matches += 1

            score = (2 if has_date else 0) + matches  # pondère la présence de "Date"
            if score > score_row:
                score_row = score
                header_row = i

        # si on n'a rien trouvé de convaincant, on tente quand même la ligne 0
        if header_row is None:
            header_row = 0

        # recharge cette feuille avec l'entête détectée
        df = pd.read_excel(excel_file, sheet_name=sheet, header=header_row)
        # drop colonnes entièrement vides
        df = df.dropna(axis=1, how="all")

        # détecte la colonne date
        date_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("date", "dates"):
                date_col = c
                break

        # si pas de colonne 'Date', on tente la première colonne
        if date_col is None and len(df.columns) > 0:
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
            except Exception:
                # pas exploitable → on saute
                continue
        else:
            # colonne date explicite
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            except Exception:
                continue

        # score “utilité” = nb de colonnes qui appartiennent à notre univers
        present = [c for c in SECURITIES if c in df.columns]
        score_useful = len(present) + score_row

        if score_useful > best_score:
            best_score = score_useful
            best_df = df.copy()
            best_info = (sheet, header_row, present)

    if best_df is None or best_df.empty:
        return pd.DataFrame()

    # tri par date, cast numérique
    best_df = best_df.sort_index()
    for c in best_df.columns:
        best_df[c] = pd.to_numeric(best_df[c], errors="coerce")

    # petit message debug utile
    if best_info:
        sheet, header_row, present = best_info
        st.caption(f"Feuille détectée: **{sheet}** | Ligne d'entêtes: **{header_row+1}** | Séries trouvées: {len(present)}")

    return best_df


def _plot_interactive(df_ind, title):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.48, 0.17, 0.17, 0.18],
        subplot_titles=(f"{title} – Price & Bollinger", "RSI", "CCI", "Z-Score")
    )

    # Price + Bands
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["price"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["sma"], name="SMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["upper"], name="Upper Band", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["lower"], name="Lower Band", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_ind.index, df_ind.index[::-1]]),
        y=pd.concat([df_ind["upper"], df_ind["lower"][::-1]]),
        fill="toself", name="Band",
        opacity=0.15, line=dict(width=0)
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["rsi"], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # CCI
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["cci"], name="CCI"), row=3, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=3, col=1)

    # Z-score
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["z"], name="Z-Score"), row=4, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=-1, line_dash="dash", line_color="green", row=4, col=1)

    fig.update_layout(height=900, legend_traceorder="normal", margin=dict(l=40, r=20, t=60, b=40))
    return fig

def render():
    st.title("Technical Analysis (Interactif)")
    uploaded = st.file_uploader("Charger un fichier Excel (sinon j’utilise `Prices/Prices sheet.xlsx`)", type=["xlsx"])

    df_raw = _load_prices(uploaded)
    if df_raw.empty:
        st.warning("Le fichier de prix est vide ou non reconnu.")
        return

    # On ne garde que les colonnes présentes dans l’univers
    cols = [c for c in SECURITIES if c in df_raw.columns]
    if not cols:
        st.warning("Aucune des séries attendues n’a été trouvée dans le fichier.")
        st.write("Colonnes détectées :", list(df_raw.columns))
        return

    left, right = st.columns([2, 1])
    with left:
        selection = st.multiselect("Séries à afficher", options=cols, default=[cols[0]])
    with right:
        min_d, max_d = df_raw.index.min(), df_raw.index.max()
        date_range = st.date_input("Période", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df_raw.loc[(df_raw.index >= start) & (df_raw.index <= end)]
        else:
            df = df_raw.copy()

    if df.empty:
        st.info("Pas de données dans cette période.")
        return

    for sec in selection:
        series = pd.to_numeric(df[sec], errors="coerce").dropna()
        if len(series) < 40:
            st.subheader(sec)
            st.info("Pas assez d’historique pour calculer les indicateurs (>= 40 points requis).")
            continue

        df_ind = _compute_indicators(series)
        last = df_ind.iloc[-1]
        signals = _analyze_signals(last)

        # Bandeau de signaux
        st.markdown(f"### {sec}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bollinger", signals["Bollinger"])
        c2.metric("RSI", signals["RSI"])
        c3.metric("CCI", signals["CCI"])
        c4.metric("Z-Score", signals["Z-Score"])

        # Ligne de tags colorés (optionnel, joli)
        st.write(
            f"- **Bollinger**: :{_signal_color(signals['Bollinger'])}[{signals['Bollinger']}]  "
            f"- **RSI**: :{_signal_color(signals['RSI'])}[{signals['RSI']}]  "
            f"- **CCI**: :{_signal_color(signals['CCI'])}[{signals['CCI']}]  "
            f"- **Z**: :{_signal_color(signals['Z-Score'])}[{signals['Z-Score']}]"
        )

        fig = _plot_interactive(df_ind, sec)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Comment lire les indicateurs ?"):
        st.markdown("""
- **Bollinger** : Prix > bande sup. = surachat (Sell) ; Prix < bande inf. = survente (Buy)  
- **RSI** : > 70 = Sell ; < 30 = Buy  
- **CCI** : > 100 = Sell ; < -100 = Buy  
- **Z-Score** : > 1 = Sell ; < -1 = Buy
        """)
