# app_key_locations.py
# --------------------
# Streamlit >= 1.25
# pip install streamlit python-dotenv lxml pandas matplotlib

import os
import io
import base64
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import plotly.graph_objects as go  # NEW


def _canon(s: str) -> str:
    """Normalise une chaîne (accents -> ASCII, casse insensible, trim)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold().strip()


CANON_PREFERRED = {
    _canon("Turkey"):  "Turkey",
    _canon("Turkiye"): "Turkey",
    _canon("Türkiye"): "Turkey",
    # (ajouter d’autres équivalences si besoin)
}


def harmonize_country_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes pays vers une forme préférée (ex: Turkiye -> Turkey)."""
    ren = {}
    for col in df.columns:
        c = _canon(col)
        pref = CANON_PREFERRED.get(c)
        if pref and col != pref:
            ren[col] = pref
    return df.rename(columns=ren)


def _get_secret(name: str) -> str | None:
    """Try st.secrets, then environment, then code fallback."""
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return v or os.getenv(name) or PL_FALLBACK.get(name)


PL_FALLBACK = {
    "PL_USERNAME": "cgregoire_http_ARo30SE9a8nb",
    "PL_PASSWORD": "X6JE22K23hIUZIaJ26IFO2senVxF41xP",
    "PL_API_KEY":  "uiqt3492x3k80mq6fu1197qm",
    "PL_API_HASH": "1EnaDpxpWnzaSpWK7BZyrsV919UmrTb1kouzooeG3rhabeDHPMteQV3jsaIQh6v3",
}

for k, v in PL_FALLBACK.items():
    os.environ.setdefault(k, v)

# Palette tab10 (bleu, orange, vert, rouge, …)
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Couleur uniformisée pour les catégories “unknown”
COLOR_MAP = {
    "Other": "#7f7f7f",
    "Not Known": "#7f7f7f",
    "Unknown": "#7f7f7f",
    "Unknown Asia": "#7f7f7f",  # utile pour l’onglet Iran
}


# --- (Optionnel) charger un .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# === Variables d'environnement à définir ===
API_URL = "https://secure.petro-logistics.com/api/v3/movementsdata"


# === Mapping des jeux de données par "location" ===
QUERY_BY_LOCATION = {
    "Global Russian exports": "FSU_FO_2023_P31",
    "Iranian exports": "Iran_FO_2023_P5",
    "Iraqi exports": "Iraq_FO_2023_P5",
    "Mexican exports": "Mexico_FO_2023_P5",
    "Venezuelan exports": "Venezuela_FO_2023_P5",
}


# === Pays par défaut (à ajuster si besoin) ===
DEFAULT_COUNTRIES = {
    "Global Russian exports": [
        "India", "China", "Egypt", "Saudi Arabia",
        "Malaysia", "Singapore", "United Arab Emirates",
        "Turkey", "Greece", "Not Known"
    ],

    # 🔹 Iran — liste ciblée (ordre préservé)
    "Iranian exports": [
        "United Arab Emirates",
        "Singapore",
        "China",
        "Malaysia",
        "Unknown Asia",
        "Floating storage",
    ],

    # 🔹 Irak — liste ciblée
    "Iraqi exports": [
        "Malaysia", "United Arab Emirates", "Egypt", "U.S.A.", "Singapore", "Not known"
    ],

    # 🔹 Mexique — point de départ pratique (à adapter)
    "Mexican exports": [
        "U.S.A.", "Panama", "Brazil", "Chile", "Peru",
        "Netherlands", "Spain", "Not known"
    ],

    # 🔹 Venezuela — point de départ pratique (à adapter)
    "Venezuelan exports": [
        "China", "Malaysia", "Singapore", "Cuba",
        "United Arab Emirates", "Greece", "Turkey",
        "Floating storage", "Not known"
    ],
}


# Pour faire correspondre les libellés “UI” aux noms réels des colonnes PL
COUNTRY_ALIASES = {
    "U.S.A.": "United States",
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "UK": "United Kingdom",
    "Turkey": "Turkiye",
    "Not Known": "Not known",   # <--- important (casse/espaces)
}


def _resolve_in_df(df_cols, requested):
    """
    Retourne (colonnes_existantes, labels_pour_legende, manquants).
    - df_cols: colonnes du DataFrame (noms de pays tels que renvoyés par l’API)
    - requested: labels “côté UI” (potentiellement aliasés)
    """
    canon_to_real = {_canon(c): c for c in df_cols}
    plot_cols, labels, missing = [], [], []

    for r in requested:
        # on essaie le libellé demandé et son alias éventuel
        candidates = [r, COUNTRY_ALIASES.get(r, r)]
        match = None
        for cand in candidates:
            real = canon_to_real.get(_canon(cand))
            if real:
                match = real
                break
        if match:
            plot_cols.append(match)  # vrai nom de colonne (API)
            labels.append(r)         # mais on affiche le label demandé
        else:
            missing.append(r)

    return plot_cols, labels, missing


# === Utils ===
def auth_header(user, pwd):
    token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

def fetch_xml(query_name: str) -> ET.Element:
    # Lire d'abord dans st.secrets, sinon variables d'environnement
    user = _get_secret("PL_USERNAME")
    pwd  = _get_secret("PL_PASSWORD")
    key  = _get_secret("PL_API_KEY")
    hsh  = _get_secret("PL_API_HASH")

    if not all([user, pwd, key, hsh]):
        raise RuntimeError(
            "Identifiants API manquants. Vérifie .streamlit/secrets.toml (PL_USERNAME, "
            "PL_PASSWORD, PL_API_KEY, PL_API_HASH) ou tes variables d'environnement."
        )

    headers = {"Accept": "application/xml", **auth_header(user, pwd)}
    payload = {"api_key": key, "api_hash": hsh, "format": "xml", "query_name": query_name}
    r = requests.post(API_URL, headers=headers, data=payload, timeout=60)
    r.raise_for_status()
    return ET.fromstring(r.text)


def compute_monthly(root: ET.Element, countries: list, since=datetime(2024,1,1)):
    monthly_sel = defaultdict(lambda: defaultdict(float))
    monthly_total = defaultdict(float)

    def label_for(api_country: str) -> str | None:
        c_api = _canon(api_country)
        for r in countries:
            if c_api == _canon(r) or c_api == _canon(COUNTRY_ALIASES.get(r, r)):
                return r
        return None

    for m in root.findall(".//movement"):
        try:
            api_country = (m.findtext("discharge_country") or "").strip()
            tonnes = float(m.findtext("qty_tonnes") or 0) / 1000.0
            date = datetime.strptime(m.findtext("load_port_date"), "%Y-%m-%d")
            if date < since:
                continue
            key = date.strftime("%Y-%m")

            monthly_total[key] += tonnes
            rlabel = label_for(api_country)
            if rlabel is not None:
                monthly_sel[key][rlabel] += tonnes
        except Exception:
            continue

    df_sel = pd.DataFrame.from_dict(monthly_sel, orient="index").fillna(0).sort_index()
    s_total = pd.Series(monthly_total, name="Total_all").sort_index()

    df = df_sel.reindex(sorted(set(df_sel.index).union(s_total.index))).fillna(0)
    df["Total_all"] = s_total.reindex(df.index).fillna(0)
    df["3M Avg"] = df["Total_all"].rolling(3).mean()
    df["12M Avg"] = df["Total_all"].rolling(12).mean()

    # plus de colonne "Other"
    return df


def compute_discharge_table(root: ET.Element, target_year: int, target_month: int):
    # ⚠️ désormais on filtre sur LOAD (exports)
    prefix = f"{target_year}-{target_month:02}"
    rows = []
    for m in root.findall(".//movement"):
        ldate = m.findtext("load_port_date")          # <--- CHANGE
        if ldate and ldate.startswith(prefix):        # <--- CHANGE
            rows.append([
                m.findtext("tanker_name"),
                m.findtext("load_port_date"),
                m.findtext("load_port"),
                m.findtext("load_country"),
                float(m.findtext("qty_tonnes") or 0),
                m.findtext("discharge_port_date"),
                m.findtext("discharge_port"),
                m.findtext("discharge_country"),
                m.findtext("cargo_id")
            ])
    cols = ["Tanker","Load Date","Load Port","Load Country","Qty Tonnes",
            "Discharge Date","Discharge Port","Discharge Country","Cargo ID"]
    return pd.DataFrame(rows, columns=cols)


def compute_yoy(root: ET.Element, countries: list, target_year: int, target_month: int):
    yoy_data = defaultdict(lambda: {"prev": 0.0, "curr": 0.0})

    def label_for(api_country: str) -> str | None:
        # renvoie le label demandé correspondant au nom API rencontré
        for r in countries:
            if api_country == r or api_country == COUNTRY_ALIASES.get(r):
                return r
        return None

    for m in root.findall(".//movement"):
        try:
            api_c = m.findtext("discharge_country")
            tonnes = float(m.findtext("qty_tonnes") or 0)
            d = datetime.strptime(m.findtext("load_port_date"), "%Y-%m-%d")
            if d.month != target_month:
                continue
            rlabel = label_for(api_c)
            if rlabel is None:
                continue
            if d.year == target_year - 1:
                yoy_data[rlabel]["prev"] += tonnes / 1000.0
            elif d.year == target_year:
                yoy_data[rlabel]["curr"] += tonnes / 1000.0
        except Exception:
            continue

    rows, tot_prev, tot_curr = [], 0.0, 0.0
    for r in countries:
        p = yoy_data[r]["prev"]; c = yoy_data[r]["curr"]
        tot_prev += p; tot_curr += c
        change = "N/A" if p == 0 else round(((c - p) / p) * 100, 1)
        rows.append([r, round(p, 2), round(c, 2), change])
    total_change = "N/A" if tot_prev == 0 else round(((tot_curr - tot_prev)/tot_prev)*100, 1)
    rows.append(["Total", round(tot_prev, 2), round(tot_curr, 2), total_change])
    return pd.DataFrame(rows, columns=["Country", f"{target_year-1} (kt)", f"{target_year} (kt)", "YoY Change (%)"])


def yoy_narrative(df_yoy: pd.DataFrame, target_year: int, target_month: int):
    month_name = datetime(target_year, target_month, 1).strftime("%B %Y")
    if df_yoy.empty or "YoY Change (%)" not in df_yoy.columns:
        return f"No YoY data available for {month_name}."
    total_row = df_yoy[df_yoy["Country"]=="Total"].iloc[0]
    change = total_row["YoY Change (%)"]
    if change == "N/A":
        return f"Year-over-year comparison for {month_name} is not available due to missing baseline."
    change_val = float(change)
    head = (f"Total exports to the selected countries "
            f"{'increased' if change_val>0 else 'decreased' if change_val<0 else 'were unchanged'} "
            f"by {abs(change_val):.1f}% vs {target_year-1}.")
    df_det = df_yoy[(df_yoy["Country"]!="Total") & (df_yoy["YoY Change (%)"]!="N/A")]
    tail = ""
    if not df_det.empty:
        srt = df_det.sort_values("YoY Change (%)", ascending=False)
        inc = srt.iloc[0]
        dec = srt.iloc[-1]
        if isinstance(inc["YoY Change (%)"], (int,float)) and inc["YoY Change (%)"]>0:
            tail += f" Largest increase: {inc['Country']} (+{inc['YoY Change (%)']:.1f}%)."
        if isinstance(dec["YoY Change (%)"], (int,float)) and dec["YoY Change (%)"]<0:
            tail += f" Largest decline: {dec['Country']} ({dec['YoY Change (%)']:.1f}%)."
    return head + tail

def style_yoy_table(df: pd.DataFrame):
    def colorize(v):
        try:
            if v == "N/A":
                return ""
            v = float(v)
            return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
        except Exception:
            return ""
    # formatage des colonnes numériques
    fmt = {df.columns[1]: "{:.2f}", df.columns[2]: "{:.2f}"}
    styler = df.style.format(fmt)
    return styler.applymap(colorize, subset=["YoY Change (%)"])

def fig_monthly_matplotlib(df: pd.DataFrame, countries: list, title: str):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    plot_cols, labels, _ = _resolve_in_df(df.columns, countries)

    if not plot_cols:
        ax.text(0.5, 0.5, "No data for selected countries", ha="center", va="center", transform=ax.transAxes)
        return fig

    df_plot = df[plot_cols].copy()
    df_plot.columns = labels

    colors = [COLOR_MAP.get(lbl, TAB10[i % len(TAB10)]) for i, lbl in enumerate(labels)]
    df_plot.plot(kind="bar", stacked=True, ax=ax, color=colors)

    if "3M Avg" in df.columns:
        ax.plot(df.index, df["3M Avg"], linestyle="--", label="3-Month Trend")
    if "12M Avg" in df.columns:
        ax.plot(df.index, df["12M Avg"], label="12-Month Avg")
    if "Total_all" in df.columns:
        ax.plot(df.index, df["Total_all"], linewidth=2, label="Total loads")

    ax.set_title(title)
    ax.set_ylabel("Kilotonnes (kt)")
    ax.set_xlabel("Month")
    ax.grid(True, axis="y")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig




def fig_monthly_plotly(df: pd.DataFrame, countries: list, title: str) -> go.Figure:
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    plot_cols, labels, _ = _resolve_in_df(df.columns, countries)
    if not plot_cols:
        fig.add_annotation(text="No data for selected countries",
                           showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    x = df.index.astype(str)

    for i, col in enumerate(plot_cols):
        lbl = labels[i]
        fig.add_trace(go.Bar(
            x=x, y=df[col],
            name=lbl,
            marker_color=COLOR_MAP.get(lbl, TAB10[i % len(TAB10)]),
            hovertemplate=f"<b>{lbl}</b><br>%{{x}}<br>%{{y:.1f}} kt<extra></extra>",
        ))

    if "3M Avg" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["3M Avg"], name="3-Month Trend",
                                 mode="lines", line=dict(dash="dash")))
    if "12M Avg" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["12M Avg"], name="12-Month Avg",
                                 mode="lines"))
    if "Total_all" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["Total_all"], name="Total loads",
                                 mode="lines"))

    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Month",
        yaxis_title="Kilotonnes (kt)",
        hovermode="x unified",
        legend=dict(orientation="v"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig




def df_to_png_table(df: pd.DataFrame, title: str = None, scale=(1.2,1.4)):
    fig, ax = plt.subplots(figsize=(10, 0.6 + max(len(df),1)*0.35))
    ax.axis("off")
    if title:
        ax.set_title(title, pad=12)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(*scale)
    plt.tight_layout()
    return fig

def make_download_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def b64_download_link(data: bytes, filename: str, label: str):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# === Rendu Streamlit de l’onglet ===
def render_key_locations_export():
    st.header("Key locations export")

    location = st.selectbox(
        "Location",
        ["Global Russian exports", "Iranian exports", "Iraqi exports"],
        index=0
    )
    query_name = QUERY_BY_LOCATION[location]
    if isinstance(query_name, str) and query_name.startswith("<") and query_name.endswith(">"):
        st.info(f"⚠️ Renseigne le 'query_name' Petro-Logistics pour **{location}** dans QUERY_BY_LOCATION.")


    colA, colB = st.columns(2)
    with colA:
        target_year = st.number_input("Target year", min_value=2020, max_value=2100,
                                      value=datetime.now().year, step=1)
    with colB:
        # par défaut: mois précédent (souvent mieux pour données stabilisées)
        default_month = max(1, datetime.now().month-1)
        target_month = st.number_input("Target month", min_value=1, max_value=12,
                                       value=default_month, step=1)

    # Pays
    default_cands = DEFAULT_COUNTRIES.get(location, [])
    countries = st.multiselect("Countries (discharge)", options=sorted(set(default_cands)),
                               default=default_cands)
    auto_top = st.checkbox("Auto-select top 10 countries since 2024 (override)", value=False)

    run = st.button("Run analysis", type="primary")
    if not run:
        return

    with st.spinner("Fetching & computing..."):
        root = fetch_xml(query_name)

        # Option top 10 auto
        if auto_top:
            cnt = defaultdict(float)
            for m in root.findall(".//movement"):
                try:
                    c = m.findtext("discharge_country")
                    t = float(m.findtext("qty_tonnes") or 0)
                    d = datetime.strptime(m.findtext("load_port_date"), "%Y-%m-%d")
                    if d >= datetime(2024,1,1) and c:
                        cnt[c] += t
                except Exception:
                    continue
            countries = [c for c,_ in sorted(cnt.items(), key=lambda x:x[1], reverse=True)[:10]]

        if not countries:
            st.error("Choisis au moins un pays.")
            return

        # Calculs
        df_month = compute_monthly(root, countries)
        df_month = harmonize_country_cols(df_month)  # <-- AJOUT

        df_dis = compute_discharge_table(root, target_year, target_month)
        df_yoy = compute_yoy(root, countries, target_year, target_month)
        yoy_text = yoy_narrative(df_yoy, target_year, target_month)  # <-- FIX

    # === Affichage ===
    st.subheader("Monthly exports")
    title = f"Monthly Fuel Oil Exports by Country — {location} (kt) — Since Jan 2024"

    # ➜ Graphe interactif (Plotly)
    fig_inter = fig_monthly_plotly(df_month, countries, title)
    st.plotly_chart(fig_inter, use_container_width=True)

    # ➜ Graphe statique (Matplotlib) pour l’export PNG
    fig_static = fig_monthly_matplotlib(df_month, countries, title)


    st.subheader("Ship tracking — exports by load month")
    st.dataframe(df_dis, use_container_width=True)
    st.download_button(
        "Download discharge CSV",
        df_dis.to_csv(index=False).encode(),
        file_name=f"discharge_{location}_{target_year}-{target_month:02}.csv",
        mime="text/csv"
    )

    st.subheader(f"YoY — {datetime(target_year, target_month,1):%B} {target_year} vs {target_year-1}")
    st.dataframe(style_yoy_table(df_yoy), use_container_width=True)

    st.download_button(
        "Download YoY CSV",
        df_yoy.to_csv(index=False).encode(),
        file_name=f"yoy_{location}_{target_year}-{target_month:02}.csv",
        mime="text/csv"
    )
    st.info(yoy_text)

    # Exports PNG (optionnels)
    st.markdown("**PNG exports (optionnels)**")
    png_monthly = make_download_png(fig_static)
    b64_download_link(png_monthly, f"monthly_{location}_{target_year}-{target_month:02}.png",
                      "Download monthly chart (PNG)")

    fig_yoy_tbl = df_to_png_table(df_yoy, title=f"YoY — {datetime(target_year, target_month,1):%B} {target_year} vs {target_year-1}")
    png_yoy = make_download_png(fig_yoy_tbl)
    b64_download_link(png_yoy, f"yoy_{location}_{target_year}-{target_month:02}.png",
                      "Download YoY table (PNG)")

    if not df_dis.empty:
        fig_dis_tbl = df_to_png_table(df_dis, title=f"Discharges {target_year}-{target_month:02}", scale=(1.0,1.2))
        png_dis = make_download_png(fig_dis_tbl)
        b64_download_link(png_dis, f"discharge_{location}_{target_year}-{target_month:02}.png",
                          "Download discharge table (PNG)")

# === Lancer comme app Streamlit si exécuté directement ===
if __name__ == "__main__":
    st.set_page_config(page_title="Key locations export", layout="wide")
    tabs = st.tabs(["Key locations export"])
    with tabs[0]:
        render_key_locations_export()
