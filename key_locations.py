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

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- (Optionnel) charger un .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# === Variables d'environnement à définir ===
API_URL = "https://secure.petro-logistics.com/api/v3/movementsdata"


# === Mapping des jeux de données par "location" ===
# ⚠️ Remplace <QUERY_IRAN>/<QUERY_IRAQ> par les bons query_name Petro-Logistics
QUERY_BY_LOCATION = {
    "Global Russian exports": "FSU_FO_2023_P31",
    "Iranian exports": "Iran_FO_2023_P5",
    "Iraqi exports": "Iraq_FO_2023_P5",
}


# === Pays par défaut (à ajuster si besoin) ===
DEFAULT_COUNTRIES = {
    # on garde la Russie comme avant
    "Global Russian exports": ["India", "China", "Egypt", "Saudi Arabia", "Malaysia", "Singapore"],

    # 🔹 Iran — on cible exactement cette liste (l’ordre est respecté)
    "Iranian exports": [
        "United Arab Emirates",
        "Singapore",
        "China",
        "Malaysia",
        "Unknown Asia",
        "Floating storage",
    ],

    # 🔹 Irak — on cible exactement cette liste
    "Iraqi exports": ["Malaysia", "United Arab Emirates", "Egypt", "U.S.A."],
}

# Pour faire correspondre les libellés "côté utilisateur" aux noms réels des colonnes PL
COUNTRY_ALIASES = {
    "U.S.A.": "United States",
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "UK": "United Kingdom",
    "Turkey": "Türkiye",
}

def _resolve_in_df(df_cols, requested):
    """Retourne (colonnes_existantes, labels_pour_legende, manquants)."""
    cols = set(df_cols)
    plot_cols, labels, missing = [], [], []
    for r in requested:
        col = r if r in cols else COUNTRY_ALIASES.get(r)
        if col in cols:
            plot_cols.append(col)
            labels.append(r)  # on garde le libellé demandé dans la légende
        else:
            missing.append(r)
    return plot_cols, labels, missing



# === Utils ===
def auth_header(user, pwd):
    token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

def fetch_xml(query_name: str) -> ET.Element:
    # Read credentials NOW (not at import time)
    user = os.getenv("PL_USERNAME")
    pwd  = os.getenv("PL_PASSWORD")
    key  = os.getenv("PL_API_KEY")
    hsh  = os.getenv("PL_API_HASH")

    if not all([user, pwd, key, hsh]):
        raise RuntimeError(
            "Identifiants API manquants. Renseigne PL_USERNAME, PL_PASSWORD, PL_API_KEY, PL_API_HASH."
        )

    headers = {"Accept": "application/xml", **auth_header(user, pwd)}
    payload = {"api_key": key, "api_hash": hsh, "format": "xml", "query_name": query_name}
    r = requests.post(API_URL, headers=headers, data=payload, timeout=60)
    r.raise_for_status()
    return ET.fromstring(r.text)


def compute_monthly(root: ET.Element, countries: list, since=datetime(2024,1,1)):
    monthly = defaultdict(lambda: defaultdict(float))
    for m in root.findall(".//movement"):
        try:
            country = m.findtext("discharge_country")
            tonnes = float(m.findtext("qty_tonnes") or 0)
            date = datetime.strptime(m.findtext("load_port_date"), "%Y-%m-%d")
            if country in countries and date >= since:
                key = date.strftime("%Y-%m")
                monthly[key][country] += tonnes / 1000.0
        except Exception:
            continue
    df = pd.DataFrame.from_dict(monthly, orient="index").fillna(0).sort_index()
    if not df.empty:
        df["Total"] = df.sum(axis=1)
        df["3M Avg"] = df["Total"].rolling(3).mean()
        df["12M Avg"] = df["Total"].rolling(12).mean()
    return df

def compute_discharge_table(root: ET.Element, target_year: int, target_month: int):
    prefix = f"{target_year}-{target_month:02}"
    rows = []
    for m in root.findall(".//movement"):
        ddate = m.findtext("discharge_port_date")
        if ddate and ddate.startswith(prefix):
            rows.append([
                m.findtext("tanker_name"),
                m.findtext("load_port_date"),
                m.findtext("load_port"),
                m.findtext("load_country"),
                float(m.findtext("qty_tonnes") or 0),
                ddate,
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

def fig_monthly(df: pd.DataFrame, countries: list, title: str):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    plot_cols, labels, missing = _resolve_in_df(df.columns, countries)

    if not plot_cols:
        ax.text(0.5, 0.5, "No data for selected countries", ha="center", va="center")
        try:
            st.warning("No matching countries found for your selection "
                       "(ex: U.S.A. → United States, UAE → United Arab Emirates).")
        except Exception:
            pass
        return fig

    df_plot = df[plot_cols].copy()
    df_plot.columns = labels
    df_plot.plot(kind="bar", stacked=True, ax=ax)

    if "3M Avg" in df.columns:
        ax.plot(df["3M Avg"], linestyle="--", label="3-Month Trend")
    if "12M Avg" in df.columns:
        ax.plot(df["12M Avg"], label="12-Month Avg")

    ax.set_title(title)
    ax.set_ylabel("Kilotonnes (kt)")
    ax.set_xlabel("Month")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y")
    ax.legend()
    plt.tight_layout()

    if missing:
        try:
            st.info("Not found in dataset: " + ", ".join(missing))
        except Exception:
            pass

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
        df_dis = compute_discharge_table(root, target_year, target_month)
        df_yoy = compute_yoy(root, countries, target_year, target_month)
        yoy_text = yoy_narrative(df_yoy, target_year, target_month)

    # === Affichage ===
    st.subheader("Monthly exports")
    title = f"Monthly Fuel Oil Exports by Country — {location} (kt) — Since Jan 2024"
    fig1 = fig_monthly(df_month, countries, title)
    st.pyplot(fig1, clear_figure=True)

    st.subheader("Discharge table")
    st.dataframe(df_dis, use_container_width=True)
    st.download_button(
        "Download discharge CSV",
        df_dis.to_csv(index=False).encode(),
        file_name=f"discharge_{location}_{target_year}-{target_month:02}.csv",
        mime="text/csv"
    )

    st.subheader(f"YoY — {datetime(target_year, target_month,1):%B} {target_year} vs {target_year-1}")
    st.dataframe(df_yoy, use_container_width=True)
    st.download_button(
        "Download YoY CSV",
        df_yoy.to_csv(index=False).encode(),
        file_name=f"yoy_{location}_{target_year}-{target_month:02}.csv",
        mime="text/csv"
    )
    st.info(yoy_text)

    # Exports PNG (optionnels)
    st.markdown("**PNG exports (optionnels)**")
    png_monthly = make_download_png(fig1)
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
