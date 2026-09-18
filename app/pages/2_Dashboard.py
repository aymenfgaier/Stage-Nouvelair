"""
Page Streamlit - Dashboard, structure en trois onglets (Sprint 3) :
Carburant & Avion, Reseau & Routes, Operationnel & Predictif.

Filtres : une periode (date range) globale s'applique aux trois onglets, plus
un filtre type d'avion (onglet Carburant & Avion) et un filtre route (onglet
Reseau & Routes), specifiques a chaque onglet.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ingestion import DB_PATH  # noqa: E402

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

if not DB_PATH.exists():
    st.warning("Aucune donnée centralisée. Passez d'abord par la page « Import & Centralisation ».")
    st.stop()

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql("SELECT * FROM flights", conn)

    def _read_optional_table(name: str) -> pd.DataFrame | None:
        try:
            return pd.read_sql(f"SELECT * FROM {name}", conn)
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            return None

    anomalies_df = _read_optional_table("anomaly_scores")
    if anomalies_df is not None and "alerte" in anomalies_df.columns:
        # SQLite n'a pas de type booleen natif : la colonne revient en 0/1 (int64) via
        # pandas.read_sql, ce qui casse le filtrage booleen direct (df[df["alerte"]])
        # sans cast explicite.
        anomalies_df["alerte"] = anomalies_df["alerte"].astype(bool)
    predictions_df = _read_optional_table("predictions")

if df.empty:
    st.warning("La base est vide.")
    st.stop()

df["date_operation_norm"] = pd.to_datetime(df["date_operation_norm"], errors="coerce")
if predictions_df is not None and "date_operation_norm" in predictions_df.columns:
    predictions_df["date_operation_norm"] = pd.to_datetime(predictions_df["date_operation_norm"], errors="coerce")

# --------------------------------------------------------------------------
# Filtre global : periode (s'applique aux trois onglets)
# --------------------------------------------------------------------------
date_min = df["date_operation_norm"].min().date()
date_max = df["date_operation_norm"].max().date()
periode = st.date_input(
    "📅 Période (filtre global, s'applique aux trois onglets)",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)
if isinstance(periode, tuple) and len(periode) == 2:
    debut, fin = periode
else:
    debut, fin = date_min, date_max

df = df[(df["date_operation_norm"].dt.date >= debut) & (df["date_operation_norm"].dt.date <= fin)]
if predictions_df is not None and "date_operation_norm" in predictions_df.columns:
    predictions_df = predictions_df[
        (predictions_df["date_operation_norm"].dt.date >= debut) & (predictions_df["date_operation_norm"].dt.date <= fin)
    ]

st.caption(f"{len(df):,}".replace(",", " ") + " vols dans la période sélectionnée.")

if df.empty:
    st.warning("Aucun vol dans la période sélectionnée.")
    st.stop()

tab_carburant, tab_routes, tab_predictif = st.tabs(["⛽ Carburant & Avion", "🗺️ Réseau & Routes", "🔮 Opérationnel & Prédictif"])

# --------------------------------------------------------------------------
# Onglet 1 : Carburant & Avion
# --------------------------------------------------------------------------
with tab_carburant:
    ac_types = sorted(df["ac_type"].dropna().unique().tolist())
    ac_selection = st.multiselect("✈️ Type d'avion", options=ac_types, default=[], placeholder="Tous les types (aucune sélection = tous)")
    df_carburant = df[df["ac_type"].isin(ac_selection)] if ac_selection else df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vols centralisés", f"{len(df_carburant):,}".replace(",", " "))
    c2.metric("Fuel réel total (t)", f"{df_carburant['fuel_reel_consomme'].sum():,.0f}".replace(",", " "))
    c3.metric("Fuel réel moyen / vol (t)", f"{df_carburant['fuel_reel_consomme'].mean():.2f}")
    c4.metric("Écart carburant moyen (t)", f"{df_carburant['ecart_carburant'].mean():+.3f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Consommation moyenne par type d'avion")
        by_fleet = df_carburant.groupby("ac_type")["fuel_reel_consomme"].mean().sort_values(ascending=False)
        st.bar_chart(by_fleet)
    with col_b:
        st.subheader("Écart carburant (réel − planifié) — distribution")
        st.bar_chart(df_carburant["ecart_carburant"].dropna().round(1).value_counts().sort_index())

    st.subheader("Consommation par mois")
    mois_labels = {
        1: "01-Jan", 2: "02-Fév", 3: "03-Mar", 4: "04-Avr", 5: "05-Mai", 6: "06-Juin",
        7: "07-Juil", 8: "08-Août", 9: "09-Sep", 10: "10-Oct", 11: "11-Nov", 12: "12-Déc",
    }
    by_month = df_carburant.groupby("mois")["fuel_reel_consomme"].sum().sort_index()
    by_month.index = by_month.index.map(mois_labels)
    st.bar_chart(by_month)

    st.divider()
    st.subheader("🚨 Alertes de dérive par immatriculation")
    if anomalies_df is None or anomalies_df.empty:
        st.info("Aucune détection de dérive disponible pour l'instant — lancez l'entraînement sur la page « Modélisation ML ».")
    else:
        n_alerts = int(anomalies_df["alerte"].sum())
        st.metric("Avions en alerte", n_alerts)
        alertes = anomalies_df[anomalies_df["alerte"]].sort_values("score_derive", ascending=False)
        if alertes.empty:
            st.success("Aucun avion en alerte de dérive actuellement.")
        else:
            st.dataframe(
                alertes[["ac_registration", "n_vols", "score_derive", "taux_alerte_recent"]].style.format(
                    {"score_derive": "{:.3f}", "taux_alerte_recent": "{:.1%}"}
                ),
                use_container_width=True,
            )

# --------------------------------------------------------------------------
# Onglet 2 : Réseau & Routes
# --------------------------------------------------------------------------
with tab_routes:
    routes = sorted(df["route"].dropna().unique().tolist())
    route_selection = st.multiselect("🗺️ Route", options=routes, default=[], placeholder="Toutes les routes (aucune sélection = toutes)")
    df_routes = df[df["route"].isin(route_selection)] if route_selection else df

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 10 routes par consommation totale")
        by_route = df_routes.groupby("route")["fuel_reel_consomme"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(by_route)
    with col_b:
        st.subheader("Top 10 routes par volume de vols")
        by_route_count = df_routes["route"].value_counts().head(10)
        st.bar_chart(by_route_count)

    st.subheader("Détail des vols")
    st.dataframe(
        df_routes[["date_operation", "ac_registration", "ac_type", "route", "trip_fuel", "fuel_reel_consomme", "ecart_carburant", "pax"]],
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Onglet 3 : Opérationnel & Prédictif
# --------------------------------------------------------------------------
with tab_predictif:
    if predictions_df is None or predictions_df.empty:
        st.info(
            "Aucune prédiction disponible pour l'instant — lancez le mode Automatique sur la page "
            "« Prédiction » pour scorer l'ensemble de la base."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Vols scorés", f"{len(predictions_df):,}".replace(",", " "))
        c2.metric("Fuel Score moyen", f"{predictions_df['fuel_score'].mean():.0f} / 100")
        c3.metric("MAE (prédit vs réel)", f"{predictions_df['fuel_reel_consomme_ecart'].abs().mean():.3f} t")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Répartition des recommandations")
            st.bar_chart(predictions_df["recommandation"].value_counts())
        with col_b:
            st.subheader("Distribution du Fuel Score")
            st.bar_chart(predictions_df["fuel_score"].round(-1).value_counts().sort_index())

        st.subheader("Vols nécessitant un complément de carburant")
        a_completer = predictions_df[predictions_df["recommandation"] == "Complément de carburant discrétionnaire recommandé"]
        a_completer = a_completer.sort_values("ecart_vs_plan", ascending=False)
        st.dataframe(
            a_completer[
                ["flight_id", "ac_registration", "route", "fuel_planifie", "fuel_reel_consomme_predit", "ecart_vs_plan", "fuel_score"]
            ].head(50),
            use_container_width=True,
        )

    st.divider()
    st.subheader("🚨 Avions en alerte de dérive")
    if anomalies_df is not None and not anomalies_df.empty:
        alertes = anomalies_df[anomalies_df["alerte"]].sort_values("score_derive", ascending=False)
        if alertes.empty:
            st.success("Aucun avion en alerte de dérive actuellement.")
        else:
            st.dataframe(alertes[["ac_registration", "n_vols", "score_derive"]], use_container_width=True)
    else:
        st.info("Aucune détection de dérive disponible pour l'instant.")
