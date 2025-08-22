# technical_analysis_tab.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === FICHIER SOURCE FIXE ===
XLSX_PATH = Path(__file__).resolve().parent / "Prices" / "Prices sheet.xlsx"

# === Univers de séries (reprend ta liste generate_charts + qq noms de l’analyse)
TARGET_TITLES = [
    "EUR FO 3.5 FOB Rdam Swap", "Brent Frontline", "Rotterdam Gasoil 0.1%",
    "180Middle east vs 180Singap", "No6 3.0 Gulf", "Barges spot crack",
    "Barges Crack spot ratio", "Visco", "Hilo", "M1/M2 380 CST spread",
    "M1/M2 Barges spread", "M1/M2 0.5 Rotter spread", "M1/M2 0.5 Singap spread",
    "HSFO E/W M1spread", "0.5 Rotter M1", "High 5 Rotterdam", "1% FO Rotterdam",
    "Lo5", "FOGO", "0.5% East/West", "TD20 M1", "380 cracks M1",
    "3.5B M0/M1", "380 cracks vs Arab Medium", "0.5 cracks vs WTI landed",
    "0.5 Rott cracks M1", "0.5 Singap cracks",
    # + celles de l’onglet technique initial
    "3.5 Barges M1", "380 CST M1", "0.5 Singap M1", "HSFO E/W M1spread"
]

# ---------- Utils chargement ----------

@st.cache_data(show_spinner=False)
def _load_prices_prices_sheet():
    if not XLSX_PATH.exists():
        st.error(f"Fichier introuvable : {XLSX_PATH}")
        return pd.DataFrame()

    xls = pd.ExcelFile(XLSX_PATH, engine="openpyxl")
    sheet = xls.sheet_names[0]  # première feuille = là où sont les prix

    # Lecture brute pour détecter la ligne d'en-têtes réelle
    tmp = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50).fillna("")
    targets = set(t.lower() for t in TARGET_TITLES)
    header_row = None
    best_matches = -1

    for i in range(min(15, len(tmp))):
        row_vals = [str(v).strip() for v in list(tmp.iloc[i, :].values)]
        row_lower = [v.lower() for v in row_vals]
        has_date = any(v == "date" for v in row_lower)
        matches = sum(1 for v in row_lower if v in targets)
        score = (2 if has_date else 0) + matches
        if score > best_matches:
            best_matches = score
            header_row = i

    if header_row is None:
        header_row = 0  # fallback

    # Relecture propre avec la bonne ligne d'entêtes
    df = pd.read_excel(xls, sheet_name=sheet, header=header_row).dropna(axis=1, how="all")
    # colonne de date
    date_col = next((c for c in df.columns if str(c).strip().lower() in ("date", "dates")), df.columns[0])

    # index temps
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # cast numérique
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, sheet, header_row + 1  # header en base 1 pour affichage


# ---------- Indicateurs & graphes ----------

def _compute_indicators(series: pd.Series, bollinger_period: int = 20):
    z = (series - series.mean()) / (series.std() if series.std() != 0 else 1)
    series = series[z.abs() <= 3]
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
    z_score = ((series - sma) / std.replace(0, np.nan)).clip(-4, 4)

    out = pd.DataFrame({
        "price": series, "sma": sma, "upper": upper, "lower": lower,
        "rsi": rsi, "cci": cci, "z": z_score
    }).dropna()
    return out

def _analyze_signals(row):
    sig = {}
    sig["Bollinger"] = "Sell" if row["price"] > row["upper"] else ("Buy" if row["price"] < row["lower"] else "Neutral")
    sig["RSI"] = "Sell" if row["rsi"] > 70 else ("Buy" if row["rsi"] < 30 else "Neutral")
    sig["CCI"] = "Sell" if row["cci"] > 100 else ("Buy" if row["cci"] < -100 else "Neutral")
    sig["Z-Score"] = "Sell" if row["z"] > 1 else ("Buy" if row["z"] < -1 else "Neutral")
    return sig

def _plot_interactive(df_ind, title):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.48, 0.17, 0.17, 0.18],
        subplot_titles=(f"{title} – Price & Bollinger", "RSI", "CCI", "Z-Score")
    )
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["price"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["sma"], name="SMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["upper"], name="Upper Band", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["lower"], name="Lower Band", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_ind.index, df_ind.index[::-1]]),
        y=pd.concat([df_ind["upper"], df_ind["lower"][::-1]]),
        fill="toself", name="Band", opacity=0.15, line=dict(width=0)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["rsi"], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["cci"], name="CCI"), row=3, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["z"], name="Z-Score"), row=4, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=-1, line_dash="dash", line_color="green", row=4, col=1)
    fig.update_layout(height=900, margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ---------- RENDER ----------

def render():
    st.title("Technical Analysis (Interactif)")
    st.caption(f"Fichier utilisé : **{XLSX_PATH}** (feuille 1)")

    df_raw, sheet, header_row_display = _load_prices_prices_sheet()
    if df_raw.empty:
        st.warning("Le fichier de prix est vide ou mal formaté.")
        return
    st.caption(f"Feuille détectée : **{sheet}** | Ligne d’entêtes : **{header_row_display}**")

    # Séries disponibles = intersection avec notre univers ; sinon on prend toutes les colonnes numériques
    cols = [c for c in df_raw.columns if c in TARGET_TITLES]
    if not cols:
        cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
        if not cols:
            st.warning("Aucune série numérique détectée.")
            st.write("Colonnes du fichier :", list(df_raw.columns))
            return

    left, right = st.columns([2, 1])
    with left:
        default_list = cols[:3] if len(cols) >= 3 else [cols[0]]
        selection = st.multiselect("Séries à afficher", options=cols, default=default_list)
    with right:
        min_d, max_d = df_raw.index.min(), df_raw.index.max()
        date_range = st.date_input("Période", value=(min_d.date(), max_d.date()),
                                   min_value=min_d.date(), max_value=max_d.date())
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df_raw.loc[(df_raw.index >= start) & (df_raw.index <= end)]

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
        signals = _analyze_signals(df_ind.iloc[-1])

        st.markdown(f"### {sec}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bollinger", signals["Bollinger"])
        c2.metric("RSI", signals["RSI"])
        c3.metric("CCI", signals["CCI"])
        c4.metric("Z-Score", signals["Z-Score"])

        fig = _plot_interactive(df_ind, sec)
        st.plotly_chart(fig, use_container_width=True)
