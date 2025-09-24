import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

def plot_bunker_price_diffs(excel_path: str = None):
    # ---------- 1) Résolution robuste du chemin ----------
    if excel_path is None:
        # On essaie d'abord relatif au fichier app.py, puis au CWD
        app_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        candidates = [
            app_dir / "Bunker diff" / "Bunker prices excel.xlsx",
            Path.cwd() / "Bunker diff" / "Bunker prices excel.xlsx",
        ]
    else:
        candidates = [Path(excel_path)]

    excel_file = None
    for p in candidates:
        if p.exists():
            excel_file = p
            break

    if excel_file is None:
        st.error(
            "Fichier Excel introuvable.\n\n"
            + "Chemins testés :\n- " + "\n- ".join(str(p) for p in candidates)
        )
        return

    st.caption(f"📄 Source Excel: `{excel_file}`")

    # ---------- 2) Lecture + nettoyage ----------
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        st.error(f"Erreur de lecture Excel: {e}")
        return

    required_cols = {"SYMBOL", "ASSESSDATE", "VALUE"}
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Colonnes manquantes dans l'Excel: {missing_cols}")
        st.write("Colonnes disponibles:", list(df.columns))
        return

    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    if "DESCRIPTION" in df.columns:
        df["DESCRIPTION"] = df["DESCRIPTION"].astype(str).str.strip()
    else:
        df["DESCRIPTION"] = df["SYMBOL"]

    df["DATE"] = pd.to_datetime(df["ASSESSDATE"], errors="coerce")
    before_drop = len(df)
    df = df.dropna(subset=["DATE", "SYMBOL", "VALUE"])
    dropped = before_drop - len(df)
    if dropped:
        st.info(f"🧹 Lignes ignorées (DATE/SYMBOL/VALUE manquants): {dropped}")

    # ---------- 3) Paires ----------
    quote_pairs = [
        ("PUAFI00", "PUAFZ00"), ("AAFER00", "PPXDK00"), ("AARBF00", "PUMFD00"),
        ("AARKD00", "AAPJW00"), ("AARSH00", "ICLO001"), ("AARTG00", "ICLO001"),
        ("AAXWG00", "PUAFZ00"), ("AAYJJ00", "PUABC00"), ("AUAMB00", "PUMFD00"),
        ("BFDZA00", "AAPJW00"), ("MFAGD00", "PUMFD00"), ("MFLOM00", "PUMFD00"),
        ("MFRDD00", "PUMFD00"), ("MFSPE00", "AMFSA00"), ("MFZHN00", "AMFSA00"),
        ("PUAER00", "AAPJW00"), ("PUAFA00", "PUABC00"), ("PUAFI00", "PUAFZ00"),
        ("PUAXP00", "AAIDC00"), ("PUBAD00", "PUAFZ00"),
        ("PPXDK00", "AAIDC00"), ("MFFJD00", "AMFSA00"),
    ]

    required_symbols = {s for a, b in quote_pairs for s in (a, b)}
    # Quels symbols existent réellement après nettoyage ?
    symbols_present = set(df["SYMBOL"].unique()) & required_symbols
    symbols_missing = required_symbols - symbols_present
    st.caption(f"✅ Symboles trouvés: {', '.join(sorted(symbols_present)) or 'aucun'}")
    if symbols_missing:
        st.warning(f"⚠️ Symboles absents du fichier: {', '.join(sorted(symbols_missing))}")

    df_filtered = df[df["SYMBOL"].isin(required_symbols)].copy()
    if df_filtered.empty:
        st.error("Aucune donnée filtrée sur les symboles requis.")
        return

    # ---------- 4) Pivot ----------
    pivot = pd.pivot_table(
        df_filtered,
        index="DATE",
        columns="SYMBOL",
        values="VALUE",
        aggfunc="mean"
    ).sort_index()

    if "PUAFZ00" in pivot.columns:
        pivot["PUAFZ00"] = pivot["PUAFZ00"] * 6.35

    desc_map = (
        df_filtered[["SYMBOL", "DESCRIPTION"]]
        .drop_duplicates()
        .set_index("SYMBOL")["DESCRIPTION"]
        .to_dict()
    )

    # ---------- 5) Diagnostics paires manquantes ----------
    missing_pairs = [(a, b) for (a, b) in quote_pairs if a not in pivot.columns or b not in pivot.columns]
    if missing_pairs:
        st.warning(
            "Ces couples sont ignorés (au moins 1 symbole absent dans le pivot) : "
            + ", ".join([f"{a}-{b}" for a, b in missing_pairs])
        )

    # ---------- 6) Tracé ----------
    year_colors = {2025: "black", 2024: "red", 2023: "green", 2022: "#1f77b4", 2021: "orange"}
    cols = st.columns(3)
    col_idx = 0
    charts_plotted = 0

    for sym1, sym2 in quote_pairs:
        if sym1 not in pivot.columns or sym2 not in pivot.columns:
            continue

        diff = pivot[sym1] - pivot[sym2]
        if diff.dropna().empty:
            st.info(f"ℹ️ Pas de dates communes ou série vide pour {sym1}-{sym2}.")
            continue

        diff_df = diff.reset_index()
        diff_df.columns = ["DATE", "DIFF"]
        diff_df["YEAR"] = diff_df["DATE"].dt.year
        diff_df["MONTH"] = diff_df["DATE"].dt.month
        diff_df["DAY"] = diff_df["DATE"].dt.day
        diff_df["FAKE_DATE"] = pd.to_datetime(
            dict(year=2000, month=diff_df["MONTH"], day=diff_df["DAY"]),
            errors="coerce"
        )

        diff_df.sort_values(["YEAR", "FAKE_DATE"], inplace=True)
        diff_df["DIFF"] = diff_df.groupby("YEAR", group_keys=False)["DIFF"].apply(lambda g: g.interpolate())

        title = f"{desc_map.get(sym1, sym1)} vs {desc_map.get(sym2, sym2)}"
        fig = go.Figure()

        years_to_plot = sorted([y for y in diff_df["YEAR"].unique() if pd.notna(y)])
        if (sym1, sym2) in [("AARSH00", "ICLO001"), ("AARTG00", "ICLO001"), ("MFRDD00", "PUMFD00")]:
            years_to_plot = [y for y in years_to_plot if y not in [2021, 2022]]

        for year in years_to_plot:
            if year == 2020:
                continue
            ydf = diff_df[diff_df["YEAR"] == year]
            if ydf["FAKE_DATE"].notna().sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=ydf["FAKE_DATE"],
                y=ydf["DIFF"],
                mode="lines",
                name=str(year),
                line=dict(color=year_colors.get(year, "grey"))
            ))

        if 2025 in years_to_plot:
            last_2025 = diff_df[(diff_df["YEAR"] == 2025) & (diff_df["DIFF"].notna())].sort_values("FAKE_DATE")
            if not last_2025.empty:
                last_point = last_2025.iloc[-1]
                fig.add_annotation(
                    x=last_point["FAKE_DATE"], y=last_point["DIFF"],
                    text=f"{last_point['DIFF']:.0f}",
                    showarrow=True, arrowhead=2, ax=0, ay=-20,
                    font=dict(size=12, color="black", family="Arial"),
                )

        fig.update_layout(
            title=title,
            xaxis=dict(title="Month", tickformat="%b", dtick="M1",
                       range=[datetime(2000, 1, 1), datetime(2000, 12, 31)]),
            yaxis_title="Price Difference (USD)",
            legend_title="Year",
            margin=dict(l=20, r=20, t=40, b=40),
            height=350, width=350,
        )

        cols[col_idx].plotly_chart(fig, use_container_width=True, key=f"bunker_{sym1}_{sym2}_{col_idx}")
        col_idx = (col_idx + 1) % 3
        charts_plotted += 1

    if charts_plotted == 0:
        st.error("Aucun graphique tracé. Vérifie le chemin du fichier, les symboles et l'intersection de dates.")

