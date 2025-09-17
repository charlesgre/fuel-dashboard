# balances/litasco_supply_tab.py
# -*- coding: utf-8 -*-
import os, glob, sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Dossier par défaut pour les fichiers Runs (UNC)
DEFAULT_LITASCO_RUNS_DIR = r"\\gvaps1\USR6\CHGE\desktop\Fuel dashboard\Litasco balances\Litasco supply"


# ========== PARAMETRISATION REGION ==========
# Dictionnaires par région. NWE est rempli; MED est un placeholder prêt à être complété.
REFINERIES_BY_REGION = {
    "NWE": {
        "Belgium": [
            {"Refinery": "Antwerpen", "Capacity": 320, "Yields": {"VLSFO": 0.10}},
            {"Refinery": "Antwerp",   "Capacity": 362, "Yields": {"HSFO": 0.06, "VLSFO": 0.06}},
        ],
        "United Kingdom": [
            {"Refinery": "Stanlow",      "Capacity": 205, "Yields": {"LSFO": 0.03}},
            {"Refinery": "Pembroke",     "Capacity": 210, "Yields": {"LSFO": 0.05, "VGO": 0.02}},
            {"Refinery": "Lindsey",      "Capacity": 113, "Yields": {"VGO": 0.04}},
            {"Refinery": "Grangemouth",  "Capacity": 150, "Yields": {"LSSR": 0.04}},
            {"Refinery": "Eastham",      "Capacity": 20,  "Yields": {"HSSR": 0.11}},
        ],
        "Sweden": [
            {"Refinery": "Preemraff Gothenburg", "Capacity": 126, "Yields": {"VGO": 0.03, "LSSR": 0.17}},
            {"Refinery": "St1 Gothenburg A B",   "Capacity": 83,  "Yields": {"VGO": 0.01, "LSSR": 0.05}},
        ],
        "France": [
            {"Refinery": "Donges",                 "Capacity": 230, "Yields": {"VLSFO": 0.07}},
            {"Refinery": "Port-Jerome Gravenchon", "Capacity": 240, "Yields": {"HSFO": 0.01, "VGO": 0.01}},
        ],
        "Netherlands": [
            {"Refinery": "Zeeland",       "Capacity": 155, "Yields": {"VLSFO": 0.04}},
            {"Refinery": "Pernis",        "Capacity": 400, "Yields": {"VLSFO": 0.11}},
            {"Refinery": "BP Rotterdam",  "Capacity": 400, "Yields": {"VLSFO": 0.13}},
        ],
        "Denmark": [
            {"Refinery": "Frederica",     "Capacity": 75,  "Yields": {"VLSFO": 0.13}},
            {"Refinery": "Kalundborg",    "Capacity": 105, "Yields": {"LSFO": 0.12}},
        ],
        "Finland": [
            {"Refinery": "Porvoo",        "Capacity": 96,  "Yields": {"VGO": 0.03}},
        ],
        "Ireland": [
            {"Refinery": "Whitegate",     "Capacity": 75,  "Yields": {"LSSR": 0.16}},
        ],
        "Lithuania": [
            {"Refinery": "Mazeikiai",     "Capacity": 200, "Yields": {"HSFO": 0.08}},
        ],
        "Poland": [
            {"Refinery": "Gdansk",        "Capacity": 210, "Yields": {"HSFO": 0.02}},
            {"Refinery": "Plock",         "Capacity": 360, "Yields": {"HSFO": 0.02}},
        ],
    },
    "MED": {
        # 👉 à compléter plus tard (capacités & rendements). La logique ci-dessous fonctionne déjà
        # dès que tu remplis ce dict de la même manière que "NWE".
    }
}

KBD_TO_KT = 6.35
COLOR_2026 = "red"; COLOR_2025 = "black"; COLOR_2024 = "green"
OTHER_CMAP = "tab20"
BAND_START, BAND_END = 2020, 2024

def pick_runs_path(explicit_path: str, runs_dir: str, pattern: str) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    files = glob.glob(os.path.join(runs_dir, pattern))
    if not files:
        raise FileNotFoundError("Aucun fichier 'Runs' trouvé.")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def load_runs(runs_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(runs_path, sheet_name=sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")

def series_for_country(runs_df: pd.DataFrame, country: str) -> pd.Series | None:
    if country in runs_df.columns:
        return runs_df[country]
    if country == "United Kingdom" and "UK" in runs_df.columns:
        return runs_df["UK"]
    return None

def nonempty_series(s: pd.Series | None) -> bool:
    if s is None: return False
    vals = pd.to_numeric(s, errors="coerce")
    return not (vals.isna().all() or np.nan_to_num(vals.values).sum() == 0.0)

def seasonality_plot_to_file(series: pd.Series, title: str, out_path: str) -> bool:
    s = series.dropna()
    if s.empty or (s == 0).all(): return False

    df = pd.DataFrame({"kt": s})
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    ym = (df.groupby(["Year","Month"])["kt"].mean().unstack(level=0).reindex(index=range(1, 13)))

    years = [int(y) for y in ym.columns if pd.notna(y)]
    if not years: return False

    band_years = [y for y in range(BAND_START, BAND_END + 1) if y in years]
    band_min = band_max = None
    if band_years:
        band_data = ym[band_years]
        band_min = band_data.min(axis=1); band_max = band_data.max(axis=1)

    plt.figure(figsize=(7.5, 4.5))
    if band_years and band_min.notna().any() and band_max.notna().any():
        plt.fill_between(range(1, 13), band_min.values, band_max.values, alpha=0.20, label=f"{BAND_START}–{BAND_END} range")

    special = {2026: COLOR_2026, 2025: COLOR_2025, 2024: COLOR_2024}
    cmap = plt.colormaps.get_cmap(OTHER_CMAP); other_idx = 0
    for y in sorted(years):
        line = ym[y]
        if y in special:
            plt.plot(range(1,13), line.values, linewidth=2.2, label=str(y), color=special[y])
        else:
            plt.plot(range(1,13), line.values, linewidth=1.3, label=str(y), color=cmap(other_idx % cmap.N), alpha=0.45)
            other_idx += 1

    plt.title(title); plt.xlabel("Month"); plt.ylabel("kt")
    plt.xticks(range(1,13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.legend(ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    return True

def build_email_html(files_by_country: dict, files_region_totals: list, run_date: str, region_label: str,
                     thumb_width_px: int = 320, cell_padding_px: int = 6) -> str:
    def rows_of_three(paths):
        for i in range(0, len(paths), 3):
            yield paths[i:i+3]
    html = []
    html.append("<html><body style='font-family:Segoe UI,Arial,sans-serif;font-size:12pt'>")
    html.append(f"<p>Hi all,</p><p>Please find below the latest seasonality charts for <b>{region_label}</b> (run date: <b>{run_date}</b>).</p>")
    for country in sorted(files_by_country.keys()):
        imgs = [p for p in files_by_country[country] if p]
        if not imgs: continue
        html.append(f"<h2 style='margin:18px 0 8px 0'>{country}</h2>")
        html.append("<table role='presentation' cellpadding='0' cellspacing='0' border='0' style='border-collapse:collapse;'>")
        for trio in rows_of_three(imgs):
            html.append("<tr>")
            for path in trio:
                cid = os.path.basename(path)
                html.append(
                    f"<td style='padding:{cell_padding_px}px; vertical-align:top;'>"
                    f"<img src='cid:{cid}' width='{thumb_width_px}' style='display:block;border:0;outline:none;text-decoration:none;'>"
                    f"</td>"
                )
            for _ in range(3 - len(trio)): html.append(f"<td style='padding:{cell_padding_px}px;'></td>")
            html.append("</tr>")
        html.append("</table>")
    if files_region_totals:
        html.append(f"<h2 style='margin:22px 0 8px 0'>{region_label} — Totaux</h2>")
        html.append("<table role='presentation' cellpadding='0' cellspacing='0' border='0' style='border-collapse:collapse;'>")
        for trio in rows_of_three([p for _, p in sorted(files_region_totals, key=lambda x: str(x[0]).lower())]):
            html.append("<tr>")
            for path in trio:
                cid = os.path.basename(path)
                html.append(
                    f"<td style='padding:{cell_padding_px}px; vertical-align:top;'>"
                    f"<img src='cid:{cid}' width='{thumb_width_px}' style='display:block;border:0;outline:none;text-decoration:none;'>"
                    f"</td>"
                )
            for _ in range(3 - len(trio)): html.append(f"<td style='padding:{cell_padding_px}px;'></td>")
            html.append("</tr>")
        html.append("</table>")
    html.append("<p style='margin-top:18px'>Best regards,<br>(auto-generated)</p>")
    html.append("</body></html>")
    return "".join(html)

def run_litasco_supply_tab():
    st.subheader("Litasco supply (Seasonality)")

    # ---- UI
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        region = st.selectbox("Région", ["NWE", "MED"], index=0, help="MED prêt pour plus tard")
    with c2:
        runs_sheet = st.text_input("Nom de l’onglet Excel (sheet)", value="NWE" if region=="NWE" else "MED")
    with c3:
        make_email = st.checkbox("Construire l’email HTML (Outlook)", value=True)

    # --- chemins & email ---
    runs_file_explicit = st.text_input(
        "Fichier Runs (facultatif, sinon on prend le plus récent)",
        value=""  # <- vide par défaut, on utilisera runs_dir + runs_pattern
    )
    runs_dir = st.text_input(
        "Dossier Runs (recherche du dernier fichier)",
        value=DEFAULT_LITASCO_RUNS_DIR
    )
    runs_pattern = st.text_input(
        "Pattern fichier",
        value="*.xlsx"   # générique ; change si tu veux "Europe Runs Recap*.xlsx"
    )

    # Dossier de sortie : par défaut un sous-dossier 'Output' à côté des Runs
    suggested_out = os.path.join(DEFAULT_LITASCO_RUNS_DIR, "Output")
    out_root = st.text_input(
        "Dossier de sortie",
        value=suggested_out
    )

    email_to = st.text_input("Email TO", value="NKRYEZIU@litasco.com; wvonschweinitz@litasco.com")
    email_cc = st.text_input("Email CC", value="mchapovalov@litasco.com; cnorgeot@litasco.com; iivanov@litasco.com")
    subj_base = st.text_input("Sujet base", value=f"{region} Fuel Production — Seasonality Charts")

    st.markdown("---")
    go = st.button("Générer")

    if not go:
        st.info("Config ok. Clique **Générer** pour produire les graphiques et l’email.")
        return

    # ---- LOGIQUE
    try:
        runs_path = pick_runs_path(runs_file_explicit, runs_dir, runs_pattern)
    except Exception as e:
        st.error(f"Erreur Runs: {e}")
        return

    try:
        runs_df = load_runs(runs_path, runs_sheet)
    except Exception as e:
        st.error(f"Lecture Excel: {e}")
        return

    REFINERIES = REFINERIES_BY_REGION.get(region, {})
    if not REFINERIES:
        st.warning(f"Aucune raffinerie configurée pour {region} (à compléter).")
        return

    target_countries = list(REFINERIES.keys())
    all_products = sorted({prod for plist in REFINERIES.values() for r in plist for prod in r["Yields"].keys()})

    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, f"Seasonals_{region}_{today}")
    os.makedirs(out_dir, exist_ok=True)

    files_by_country = {c: [] for c in target_countries}
    files_region_totals = []
    country_product = {c: {} for c in target_countries}
    totals_product = {p: None for p in all_products}

    # Calculs
    for country in target_countries:
        runs_c = series_for_country(runs_df, country)
        if not nonempty_series(runs_c): continue

        refs = REFINERIES[country]
        total_cap = sum(r["Capacity"] for r in refs)
        if total_cap == 0: continue

        util = runs_c / total_cap  # facteur d'utilisation

        for p in all_products:
            country_product[country][p] = None

        for r in refs:
            Ri = r["Capacity"] * util
            prod_cols = {}
            for prod, y in r["Yields"].items():
                series_kt = Ri * y * KBD_TO_KT
                prod_cols[prod] = series_kt

                if country_product[country][prod] is None:
                    country_product[country][prod] = series_kt.copy()
                else:
                    country_product[country][prod] = country_product[country][prod].add(series_kt, fill_value=0.0)

                if totals_product[prod] is None:
                    totals_product[prod] = series_kt.copy()
                else:
                    totals_product[prod] = totals_product[prod].add(series_kt, fill_value=0.0)

            for p in all_products:
                if p not in prod_cols:
                    prod_cols[p] = pd.Series(0.0, index=Ri.index)

            df_ref = pd.DataFrame(prod_cols, index=Ri.index)[all_products]

            # Graphs Raffinerie × Produit
            for product in df_ref.columns:
                s = df_ref[product]
                if s.isna().all() or (s == 0).all(): continue
                fname = f"{region}_{country.replace(' ','_')}__{r['Refinery'].replace(' ','_')}__{product}_seasonal.png"
                out_path = os.path.join(out_dir, fname)
                if seasonality_plot_to_file(s, title=f"Seasonality — {region} / {country} / {r['Refinery']} / {product} (kt)", out_path=out_path):
                    files_by_country[country].append(out_path)

    # Graphs Pays × Produit
    for country, prod_map in country_product.items():
        cols = {p: s for p, s in prod_map.items() if s is not None and not (s.isna().all() or (np.nan_to_num(s.values).sum() == 0.0))}
        if not cols: continue
        df_country = pd.DataFrame(cols)
        for product in df_country.columns:
            s = df_country[product]
            if s.isna().all() or (s == 0).all(): continue
            fname = f"{region}_{country.replace(' ','_')}__{product}_TOTAL_country_seasonal.png"
            out_path = os.path.join(out_dir, fname)
            if seasonality_plot_to_file(s, title=f"Seasonality — {region} / {country} / {product} (kt)", out_path=out_path):
                files_by_country[country].append(out_path)

    # Graphs Région (totaux)
    for product, s in totals_product.items():
        if s is None or s.isna().all() or (np.nan_to_num(s.values).sum() == 0.0): continue
        fname = f"{region}_TOTAL__{product}_seasonal.png"
        out_path = os.path.join(out_dir, fname)
        if seasonality_plot_to_file(s, title=f"Seasonality — {region} Total Supply / {product} (kt)", out_path=out_path):
            files_region_totals.append((product, out_path))

    st.success(f"Graphiques enregistrés dans : {out_dir}")

    # Affichage rapide dans l’app
    with st.expander("Aperçu des images (quelques-unes)"):
        count = 0
        for country in sorted(files_by_country.keys()):
            for p in files_by_country[country][:3]:
                st.image(p, caption=os.path.basename(p), use_container_width=True)
                count += 1
                if count >= 9:
                    break
            if count >= 9: break

    # Export CSV résumé
    summary_rows = []
    for country, prod_map in country_product.items():
        for product, s in prod_map.items():
            if s is None: continue
            summary_rows.append({"Region": region, "Country": country, "Product": product,
                                "Avg_kt": round(float(s.mean()), 2), "Total_kt": round(float(s.sum()), 2)})
    for product, s in totals_product.items():
        if s is None: continue
        summary_rows.append({"Region": region, "Country": f"{region} Total", "Product": product,
                            "Avg_kt": round(float(s.mean()), 2), "Total_kt": round(float(s.sum()), 2)})

    df_summary = pd.DataFrame(summary_rows).sort_values(["Region","Country","Product"]).reset_index(drop=True)
    st.dataframe(df_summary, use_container_width=True)
    csv_path = os.path.join(out_dir, f"Summary_{region}_{today}.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    st.download_button("Télécharger le CSV", data=open(csv_path,"rb"), file_name=os.path.basename(csv_path), mime="text/csv")

    # Email (HTML + pièces jointes inline)
    if make_email:
        html = build_email_html(files_by_country, files_region_totals, today, region_label=region, thumb_width_px=400, cell_padding_px=6)
        st.download_button("Télécharger l’HTML de l’email", data=html.encode("utf-8"), file_name=f"email_{region}_{today}.html", mime="text/html")

        if st.checkbox("Ouvrir un brouillon Outlook (Windows + pywin32)", value=False):
            try:
                import win32com.client as win32
                outlook = win32.Dispatch("Outlook.Application")
                mail = outlook.CreateItem(0)
                mail.To = email_to; mail.CC = email_cc
                mail.Subject = f"{subj_base} — {today}"

                # Attacher images en inline
                cid_map = {}
                for _, paths in files_by_country.items():
                    for p in paths:
                        cid_map[os.path.basename(p)] = p
                for _, p in files_region_totals:
                    cid_map[os.path.basename(p)] = p

                for cid, path in cid_map.items():
                    attach = mail.Attachments.Add(path)
                    attach.PropertyAccessor.SetProperty("http://schemas.microsoft.com/mapi/proptag/0x3712001F", cid)

                mail.HTMLBody = html
                mail.Display(True)
                st.info("Fenêtre de composition Outlook ouverte.")
            except Exception as e:
                st.warning(f"pywin32 non disponible ou Outlook indisponible : {e}")
