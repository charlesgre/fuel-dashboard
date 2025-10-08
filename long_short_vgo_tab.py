# long_short_vgo_tab.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from calendar import monthrange
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# -------------------------
#  Config / constants
# -------------------------

# Noms fournis par toi
SWEET_REFINERIES = {
    "Antwerp", "Fredericia", "Kalundborg", "Porvoo", "Donges", "Gonfreville",
    "Hamburg (Heide)", "Whitegate", "BP Rotterdam", "Exxon Rotterdam", "Mongstad",
    "Sines", "Preemraff Lysekil", "Preemraff Gothenburg", "St1 Gothenburg A B",
    "Stanlow", "Pembroke", "Humber", "Fawley",
}
SOUR_REFINERIES = {
    "Port-Jerome Gravenchon", "Heide", "Mazeikiai", "Pernis", "Gdansk", "Plock",
    "Bilbao", "Nynas Gothenburg", "Nynashamn",
}

# Ces colonnes/labels existent dans le fichier Excel (onglet "Query1")
UNIT_TYPES_EXPECTED = [
    "Vacuum Distillation",
    "FCCU (Fluid Catalytic Cracker)",
    "Distillate Hydrocracker",
]

# Dossier & nom par défaut (à adapter si besoin)
DEFAULT_FILE_CANDIDATES = [
    Path(__file__).resolve().parent / "Long short VGO" / "long-short VGO master.xlsx",
    Path(__file__).resolve().parent / "long-short VGO master.xlsx",
]

NWE_REGION_SUBSTRING = "Northwest Europe"  # filtre par région


# -------------------------
#  Helpers
# -------------------------

def _find_excel_path(user_path: Optional[str]) -> Path:
    """Best effort pour trouver le fichier Excel."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    for cand in DEFAULT_FILE_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Fichier Excel non trouvé. Renseigne un chemin manuel dans la barre latérale."
    )


def _make_alias_map(rotterdam_target: str) -> Dict[str, str]:
    """
    rotterdam_target: "BP Rotterdam" ou "Exxon Rotterdam".
    On mappe le label générique "Rotterdam Refinery" vers la cible choisie.
    """
    assert rotterdam_target in {"BP Rotterdam", "Exxon Rotterdam"}
    return {
        "Notre-Dame-de-Gravenchon Refinery": "Port-Jerome Gravenchon",
        "St1 Gothenburg Refinery": "St1 Gothenburg A B",
        "Rotterdam Refinery": rotterdam_target,
    }


def _map_crude(plant: str, alias_map: Dict[str, str]) -> Optional[str]:
    p = alias_map.get(plant, plant)
    if any(nm.lower() in p.lower() or p.lower() in nm.lower() for nm in SWEET_REFINERIES):
        return "sweet"
    if any(nm.lower() in p.lower() or p.lower() in nm.lower() for nm in SOUR_REFINERIES):
        return "sour"
    return None


def _month_overlap_days(start: pd.Timestamp, end: pd.Timestamp, year: int, month: int) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    month_start = pd.Timestamp(year=year, month=month, day=1)
    days_in_month = monthrange(year, month)[1]
    month_end = month_start + pd.Timedelta(days=days_in_month)  # exclusive
    overlap_start = max(start, month_start)
    overlap_end = min(end, month_end)
    return max(0, (overlap_end - overlap_start).days)


def _month_online_fraction(event_row: pd.Series, year: int, month: int) -> float:
    """Retourne la fraction 'online' d'une ligne événement pour ce mois."""
    days_in_month = monthrange(year, month)[1]
    overlap = _month_overlap_days(event_row["EVENTSTARTDATE"], event_row["EVENTENDDATE"], year, month)
    if overlap <= 0:
        return 1.0
    derate_frac = float(event_row.get("DERATE", 100) or 100) / 100.0  # 100 => full down
    online = 1.0 - (overlap / days_in_month) * derate_frac
    return float(np.clip(online, 0.0, 1.0))


def _vgo_yield(crude: Optional[str]) -> float:
    if crude == "sweet":
        return 0.60
    if crude == "sour":
        return 0.30
    return np.nan


# -------------------------
#  Data loading / caching
# -------------------------

@st.cache_data(show_spinner=False)
def _load_query1(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name="Query1")
    # Nettoyage minimal
    for c in ("EVENTSTARTDATE", "EVENTENDDATE"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _compute_month_view(
    excel_path: str,
    rotterdam_target: str,
    year: int,
    month: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcule la vue mensuelle (NWE) + table TARS du mois.
    Retourne:
      - df_month: per refinery (capacités dispo après TAR, VGO Out, Demand, Balance, Status, Events JSON)
      - df_tars:  ligne par TAR chevauchant le mois (après filtres de région)
    """
    alias_map = _make_alias_map(rotterdam_target)
    df = _load_query1(excel_path)

    # Normalisation noms & crude type
    df["PLANT_ALIAS"] = df["PLANTNAME"].replace(alias_map)
    df["CRUDE_TYPE"] = df["PLANT_ALIAS"].apply(lambda x: _map_crude(x, alias_map))

    # Garder seulement unités pertinentes
    df_rel = df[df["UNITTYPEDESC"].isin(UNIT_TYPES_EXPECTED)].copy()

    # Calcul de la fraction online sur chaque ligne d'événement
    df_rel["ONLINE_FRAC"] = df_rel.apply(lambda r: _month_online_fraction(r, year, month), axis=1)

    # On veut la pire dispo par unité physique (UNITNAME)
    agg = (
        df_rel.groupby(
            ["REGION", "COUNTRY", "PLANT_ALIAS", "CRUDE_TYPE", "UNITTYPEDESC", "UNITNAME"],
            dropna=False,
        )
        .apply(lambda g: pd.Series({"CAPACITY": g["CAPACITY"].iloc[0], "ONLINE_FRAC": g["ONLINE_FRAC"].min()}))
        .reset_index()
    )
    agg["EFF_CAP"] = agg["CAPACITY"] * agg["ONLINE_FRAC"]

    # TARs par raffinerie qui chevauchent le mois
    tar_rows: List[Dict] = []
    for (reg, cnt, ref, crude), sub in df_rel.groupby(["REGION", "COUNTRY", "PLANT_ALIAS", "CRUDE_TYPE"]):
        for _, r in sub.iterrows():
            ov = _month_overlap_days(r["EVENTSTARTDATE"], r["EVENTENDDATE"], year, month)
            if ov > 0:
                tar_rows.append(
                    {
                        "REGION": reg,
                        "COUNTRY": cnt,
                        "Refinery": ref,
                        "Crude": crude,
                        "Unit": r["UNITTYPEDESC"],
                        "UnitName": r["UNITNAME"],
                        "DeratePct": int(r.get("DERATE", 100) or 100),
                        "Start": r["EVENTSTARTDATE"],
                        "End": r["EVENTENDDATE"],
                        "DaysOverlap": int(ov),
                        "EventType": r.get("EVENTTYPE", ""),
                        "Cause": r.get("EVENTCAUSE", ""),
                        "Capacity": float(r.get("CAPACITY", 0.0) or 0.0),
                    }
                )
    df_tars = pd.DataFrame(tar_rows)

    # Pivot pour total par raffinerie
    piv = (
        agg.groupby(["REGION", "COUNTRY", "PLANT_ALIAS", "CRUDE_TYPE", "UNITTYPEDESC"])["EFF_CAP"]
        .sum()
        .unstack("UNITTYPEDESC")
        .fillna(0.0)
        .reset_index()
    )
    for col in UNIT_TYPES_EXPECTED:
        if col not in piv.columns:
            piv[col] = 0.0

    piv = piv.rename(
        columns={
            "Vacuum Distillation": "Vacuum Cap (Avail)",
            "FCCU (Fluid Catalytic Cracker)": "FCCU Cap (Avail)",
            "Distillate Hydrocracker": "DHC Cap (Avail)",
        }
    )
    piv["VGO_Out"] = piv.apply(lambda r: r["Vacuum Cap (Avail)"] * _vgo_yield(r["CRUDE_TYPE"]), axis=1)
    piv["VGO_Demand"] = piv["FCCU Cap (Avail)"] + piv["DHC Cap (Avail)"]
    piv["Balance"] = piv["VGO_Out"] - piv["VGO_Demand"]
    piv["Status"] = np.where(
        (piv["FCCU Cap (Avail)"] == 0) & (piv["DHC Cap (Avail)"] == 0),
        "Long",
        np.where(piv["Balance"] >= 0, "Long", "Short"),
    )

    piv = piv.rename(columns={"PLANT_ALIAS": "Refinery", "CRUDE_TYPE": "Crude"})

    # Filtre région NWE
    df_month = piv[piv["REGION"].astype(str).str.contains(NWE_REGION_SUBSTRING, na=False)].copy()
    if not df_tars.empty:
        df_tars = df_tars[df_tars["REGION"].astype(str).str.contains(NWE_REGION_SUBSTRING, na=False)].copy()

    # Ajout du mois/année pour export
    df_month["Year"] = year
    df_month["Month"] = month
    if not df_tars.empty:
        df_tars["Year"] = year
        df_tars["Month"] = month

    return df_month.reset_index(drop=True), df_tars.reset_index(drop=True)


# -------------------------
#  UI (Streamlit)
# -------------------------

def render_long_short_vgo_tab(default_excel_path: Optional[str] = None) -> None:
    st.subheader("VGO Long / Short — NWE (TARS-aware)")

    # ---- Barre latérale
    with st.sidebar:
        st.markdown("### ⚙️ VGO — Paramètres")
        excel_path_str = st.text_input(
            "Chemin du fichier Excel (onglet Query1)",
            value=str(_find_excel_path(default_excel_path)),
            help="Laisse le chemin par défaut s'il est correct.",
        )
        rotterdam_target = st.selectbox(
            "Mapping pour 'Rotterdam Refinery'",
            options=["BP Rotterdam", "Exxon Rotterdam"],
            index=0,  # par défaut BP comme demandé
            help="Recalcule la vue mensuelle avec ce mapping d'alias.",
        )

    # ---- Période
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        year = st.number_input("Année", min_value=2022, max_value=2035, value=2026, step=1)
    with c2:
        month = st.number_input("Mois", min_value=1, max_value=12, value=2, step=1)
    with c3:
        st.caption("Sélectionne l'année et le mois – toutes les TARS chevauchant ce mois sont prises en compte.")

    # ---- Calcul
    try:
        df_month, df_tars = _compute_month_view(excel_path_str, rotterdam_target, int(year), int(month))
    except Exception as e:
        st.error("Impossible de calculer la vue mensuelle.")
        with st.expander("Traceback"):
            st.exception(e)
        return

    # ---- Filtres (Crude / Refineries)
    crude_choices = ["all", "sweet", "sour"]
    crude = st.selectbox("Crude", crude_choices, index=0, help="Filtre sur sweet/sour (selon ta liste).")

    refs = sorted(df_month["Refinery"].unique().tolist())
    selected_refs = st.multiselect("Refineries (multi)", options=refs, default=[])

    def _apply_filters(dfX: pd.DataFrame) -> pd.DataFrame:
        out = dfX.copy()
        if crude != "all":
            out = out[out["Crude"] == crude]
        if selected_refs:
            out = out[out["Refinery"].isin(selected_refs)]
        return out

    df_f = _apply_filters(df_month)
    df_tars_f = _apply_filters(df_tars) if not df_tars.empty else df_tars

    # ---- KPIs
    vgo_out = float(df_f["VGO_Out"].sum())
    vgo_dem = float(df_f["VGO_Demand"].sum())
    long_total = float(df_f.loc[df_f["Status"] == "Long", "Balance"].sum())
    short_total = float((df_f.loc[df_f["Status"] == "Short", "Balance"]).abs().sum())
    n_long = int((df_f["Status"] == "Long").sum())
    n_short = int((df_f["Status"] == "Short").sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("VGO Out (mois)", f"{vgo_out:,.0f}")
    k2.metric("VGO Demand (mois)", f"{vgo_dem:,.0f}")
    k3.metric("Long total", f"{long_total:,.0f}")
    k4.metric("Short total", f"{short_total:,.0f}")
    k5.metric("Sites Long / Short", f"{n_long} / {n_short}")

    st.markdown("---")

    # ---- Bar chart Long/Short
    if df_f.empty:
        st.info("Aucune donnée pour ce filtre.")
        return

    df_plot = df_f.copy()
    df_plot["RefLabel"] = df_plot["Refinery"] + " (" + df_plot["COUNTRY"] + ")"
    # tri: Short par besoin décroissant, puis Long par surplus décroissant
    df_plot["Need"] = np.where(df_plot["Status"] == "Short", -df_plot["Balance"], 0.0)
    df_plot["Surplus"] = np.where(df_plot["Status"] == "Long", df_plot["Balance"], 0.0)
    df_plot = df_plot.sort_values(["Status", "Need", "Surplus"], ascending=[True, False, False])

    color_map = {"Long": "#2e7d32", "Short": "#c62828"}
    fig_bar = px.bar(
        df_plot,
        x="RefLabel",
        y="Balance",
        color="Status",
        color_discrete_map=color_map,
        hover_data={
            "RefLabel": True,
            "COUNTRY": True,
            "Crude": True,
            "Vacuum Cap (Avail)": ":,.0f",
            "FCCU Cap (Avail)": ":,.0f",
            "DHC Cap (Avail)": ":,.0f",
            "VGO_Out": ":,.0f",
            "VGO_Demand": ":,.0f",
            "Balance": ":,.0f",
        },
    )
    fig_bar.update_layout(
        title=f"VGO Balance par raffinerie — {year}-{month:02d} (NWE)",
        xaxis_title=None,
        yaxis_title="Balance (unités capacité)",
        xaxis_tickangle=60,
        height=480,
        legend_title=None,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="vgo_bar")

    # ---- Waterfall régional
    wf = go.Figure(
        go.Waterfall(
            name="VGO NWE",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["VGO Out", "– FCCU", "– DHC", "Balance"],
            y=[vgo_out, -df_f["FCCU Cap (Avail)"].sum(), -df_f["DHC Cap (Avail)"].sum(), vgo_out - vgo_dem],
            connector={"line": {"color": "rgba(0,0,0,0.3)"}},
        )
    )
    wf.update_layout(title=f"Waterfall — {year}-{month:02d} (NWE, après filtres)", height=420)
    st.plotly_chart(wf, use_container_width=True, key="vgo_wf")

    # ---- Détail raffinerie (click simulé via selectbox pour rester simple)
    ref_for_detail = st.selectbox("Détail raffinerie", options=df_plot["Refinery"].tolist())
    ref_row = df_month[(df_month["Refinery"] == ref_for_detail)].iloc[0]
    st.markdown(
        f"**{ref_row['Refinery']}** — {ref_row['COUNTRY']} — "
        f"Status: **: {'🟢 Long' if ref_row['Status']=='Long' else '🔴 Short'}**  \n"
        f"Crude: **{ref_row['Crude']}** | "
        f"VDU: **{ref_row['Vacuum Cap (Avail)']:,.0f}** | "
        f"FCCU: **{ref_row['FCCU Cap (Avail)']:,.0f}** | "
        f"DHC: **{ref_row['DHC Cap (Avail)']:,.0f}** | "
        f"VGO_Out: **{ref_row['VGO_Out']:.0f}** | "
        f"Demand: **{ref_row['VGO_Demand']:.0f}** | "
        f"Balance: **{ref_row['Balance']:.0f}**"
    )

    # ---- TARs du mois (après filtres)
    st.markdown("### TARs du mois (après filtres)")
    if df_tars_f.empty:
        st.info("Aucun TAR qui chevauche le mois sélectionné pour ce filtre.")
    else:
        # tri lisible : début puis raffinerie
        df_tars_f = df_tars_f.sort_values(["Start", "Refinery", "Unit", "UnitName"]).copy()
        df_tars_f_disp = df_tars_f[
            [
                "Refinery",
                "Unit",
                "UnitName",
                "DeratePct",
                "Start",
                "End",
                "DaysOverlap",
                "EventType",
                "Cause",
                "Capacity",
                "Crude",
                "COUNTRY",
            ]
        ].rename(
            columns={
                "DeratePct": "Derate %",
                "DaysOverlap": "Days",
            }
        )
        st.dataframe(df_tars_f_disp, use_container_width=True, hide_index=True)

        # Export TARs filtrées
        csv_tars = df_tars_f_disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export TARs (mois filtré)",
            data=csv_tars,
            file_name=f"VGO_TARs_{year}_{month:02d}_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # ---- Table mois (après filtres) + Export
    st.markdown("### Table du mois (après filtres)")
    df_month_disp = df_f[
        [
            "REGION",
            "COUNTRY",
            "Refinery",
            "Crude",
            "Vacuum Cap (Avail)",
            "VGO_Out",
            "FCCU Cap (Avail)",
            "DHC Cap (Avail)",
            "VGO_Demand",
            "Balance",
            "Status",
        ]
    ].copy()
    st.dataframe(df_month_disp, use_container_width=True, hide_index=True)
    csv_month = df_month_disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export table mois (après filtres)",
        data=csv_month,
        file_name=f"VGO_Month_{year}_{month:02d}_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
