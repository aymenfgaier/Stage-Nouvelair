"""
Page Streamlit - Prediction : deux modes.

  - Manuel      : soit un vol existant est choisi (le plan de vol reel, Trip
                  Fuel/Taxi Fuel, est alors connu et utilise -> modele principal,
                  R2=0.963), soit un vol est saisi librement, hypothetique, sans
                  plan de vol -> modele dedie "sans plan" (R2=0.945).
  - Automatique : la prediction est lancee sur l'ensemble de la base de
                  donnees centralisee (jamais sur un fichier) avec le modele
                  principal, et les resultats sont enregistres dans la table
                  `predictions`.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2] / "ml"))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from datasource import database_exists, database_row_count, load_from_database  # noqa: E402
from ingestion import DB_PATH  # noqa: E402
from model_training import MODELS_DIR, predict_batch_from_database, predict_manual_with_plan, predict_manual_without_plan  # noqa: E402
from recommendation import compute_fuel_score, generate_recommendation  # noqa: E402

st.set_page_config(page_title="Prédiction", page_icon="🔮", layout="wide")
st.title("🔮 Prédiction de consommation")
st.caption("Fuel réel total — mode manuel (avec ou sans plan de vol) ou automatique")

if not database_exists():
    st.warning("Aucune donnée centralisée. Passez d'abord par la page « Import & Centralisation ».")
    st.stop()

models_ready = (MODELS_DIR / "fuel_reel_consomme_xgboost.joblib").exists() and (MODELS_DIR / "fuel_reel_consomme_manuel_xgboost.joblib").exists()
if not models_ready:
    st.warning("Le modèle n'est pas encore entraîné. Passez par la page « Modélisation ML » d'abord.")
    st.stop()

mode = st.radio("Mode de prédiction", options=["Manuel", "Automatique"], horizontal=True)

if mode == "Manuel":
    st.markdown("Choisissez ou saisissez les caractéristiques d'un vol pour obtenir une prédiction à la demande.")

    result = load_from_database()
    df = result.df
    ac_types = sorted(df["ac_type"].dropna().unique().tolist())

    # Correspondance nom complet d'aéroport <-> code ICAO (bijective dans les données)
    dep_name_to_icao = dict(zip(df["dep_airport"], df["dep_icao"]))
    dest_name_to_icao = dict(zip(df["dest_airport"], df["dest_icao"]))
    dep_names = sorted(dep_name_to_icao.keys())
    dest_names = sorted(dest_name_to_icao.keys())

    input_mode = st.radio("Origine des données saisies", ["Choisir un vol existant de la base", "Saisir manuellement"], horizontal=True)

    if input_mode == "Choisir un vol existant de la base":
        # Etape 1 : une seule case de recherche + selection, filtrage en direct cote client
        # au fur et a mesure de la frappe (comportement natif de st.selectbox), sur la liste
        # des numeros de vol EXACTS (pas les lignes brutes) pour ne jamais afficher un vol different.
        flight_ids = sorted(df["flight_id"].dropna().unique().tolist())
        flight_id = st.selectbox(
            "🔎 Numéro de vol",
            options=flight_ids,
            index=None,
            placeholder="Tapez pour rechercher, ex. BJ 0822",
        )

        if flight_id is None:
            st.info("Commencez à taper un numéro de vol, puis choisissez-le dans la liste proposée.")
            st.stop()

        # Etape 2 : un meme numero de vol revient a plusieurs dates -> on choisit l'occurrence exacte
        occurrences = df[df["flight_id"] == flight_id].sort_values("date_operation_norm", ascending=False).copy()
        occurrences["_label"] = occurrences["date_operation"].astype(str) + "  —  " + occurrences["route"].astype(str)
        selected_label = st.selectbox(f"Occurrence de {flight_id} ({len(occurrences)} vol(s))", options=occurrences["_label"].tolist())
        row = occurrences[occurrences["_label"] == selected_label].iloc[0]

        ac_type = row["ac_type"]
        dep_icao = row["dep_icao"]
        dest_icao = row["dest_icao"]
        adults = int(row["adults"])
        children = int(row["children"])
        pax = int(row["pax"])
        duree_prevue_min = float(row["duree_prevue_min"])
        mois = int(row["mois"])
        jour_semaine = int(row["jour_semaine"])
        trip_fuel = float(row["trip_fuel"])
        taxi_fuel = float(row["taxi_fuel"])

        st.markdown("**Informations du vol sélectionné**")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Date", str(row["date_operation"]))
        i2.metric("Avion", f"{row['ac_registration']} ({ac_type})")
        i3.metric("Route", f"{row['dep_airport']} → {row['dest_airport']}")
        i4.metric("Passagers", pax)
        i5, i6, i7, i8 = st.columns(4)
        i5.metric("Trip Fuel planifié", f"{trip_fuel:.2f} t")
        i6.metric("Taxi Fuel planifié", f"{taxi_fuel:.2f} t")
        i7.metric("Fuel réel observé", f"{row['fuel_reel_consomme']:.2f} t")
        ecart_planifie_reel = (trip_fuel + taxi_fuel) - float(row["fuel_reel_consomme"])
        i8.metric("Écart planifié − réel", f"{ecart_planifie_reel:+.2f} t")
        st.caption("Trip Fuel/Taxi Fuel sont affichés pour référence uniquement — ils ne sont pas utilisés par le modèle.")

        observed_fuel = float(row["fuel_reel_consomme"])
    else:
        observed_fuel = None
        c1, c2, c3 = st.columns(3)
        ac_type = c1.selectbox("Type d'avion", options=ac_types)
        dep_name = c2.selectbox("Aéroport de départ", options=dep_names)
        dest_name = c3.selectbox("Aéroport de destination", options=dest_names)
        dep_icao = dep_name_to_icao[dep_name]
        dest_icao = dest_name_to_icao[dest_name]
        c4, c5, c6 = st.columns(3)
        adults = c4.number_input("Adultes", min_value=0, max_value=250, value=110)
        children = c5.number_input("Enfants", min_value=0, max_value=100, value=10)
        pax = c6.number_input("Passagers (total)", min_value=0, max_value=250, value=120)
        c7, c8 = st.columns(2)
        duree_prevue_min = c7.number_input("Durée prévue (minutes)", min_value=10, max_value=600, value=180)
        mois = c8.selectbox("Mois", options=list(range(1, 13)), format_func=lambda m: pd.Timestamp(2025, m, 1).strftime("%B"))
        jour_semaine = st.selectbox("Jour de la semaine", options=list(range(7)), format_func=lambda d: ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"][d])

    if st.button("Prédire", type="primary"):
        if observed_fuel is not None:
            predicted = predict_manual_with_plan(
                ac_type, dep_icao, dest_icao, adults, children, pax, duree_prevue_min, mois, jour_semaine,
                trip_fuel, taxi_fuel,
            )
            c1, c2 = st.columns(2)
            c1.metric("Fuel réel total prédit", f"{predicted:.3f} t")
            c2.metric("Écart prédit − réel observé", f"{predicted - observed_fuel:+.3f} t")
            reco = generate_recommendation(predicted, trip_fuel + taxi_fuel)
            fuel_score = compute_fuel_score(observed_fuel, predicted)
        else:
            predicted = predict_manual_without_plan(ac_type, dep_icao, dest_icao, adults, children, pax, duree_prevue_min, mois, jour_semaine)
            st.metric("Fuel réel total prédit", f"{predicted:.3f} t")
            st.caption(
                "Aucune valeur réelle observée disponible pour comparaison (vol saisi manuellement, hypothétique). "
                "Prédiction obtenue avec le modèle dédié à la saisie manuelle (sans plan de vol, R²=0,945)."
            )
            reco = generate_recommendation(predicted, None)
            fuel_score = None

        st.divider()
        st.markdown("**Recommandation**")
        r1, r2 = st.columns(2)
        r1.metric("Carburant optimal proposé", f"{reco.fuel_optimal_propose:.3f} t")
        if fuel_score is not None:
            r2.metric("Fuel Score", f"{fuel_score:.0f} / 100")
        st.info(reco.message, icon="⛽")
        st.caption(f"Cost index : {reco.cost_index_suggestion}")

else:  # Automatique
    st.markdown(
        "La prédiction automatique s'exécute sur **l'ensemble du référentiel centralisé** "
        "(table `flights`), jamais sur un fichier ponctuel. Les résultats sont enregistrés "
        "dans la table `predictions` de la base."
    )
    n_rows = database_row_count()
    st.metric("Vols dans la base à scorer", f"{n_rows:,}".replace(",", " "))

    if st.button("Lancer la prédiction automatique sur la base", type="primary"):
        with st.spinner("Scoring de l'ensemble des vols centralisés..."):
            batch = predict_batch_from_database(DB_PATH)
        st.success(f"{len(batch)} vols scorés et enregistrés dans `predictions`.")

        ecart = batch["fuel_reel_consomme_ecart"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Erreur absolue moyenne (MAE)", f"{ecart.abs().mean():.3f} t")
        c2.metric("Erreur moyenne signée (biais)", f"{ecart.mean():+.3f} t")
        c3.metric("Erreur absolue max", f"{ecart.abs().max():.3f} t")
        st.caption(
            "⚠️ Le **biais** (erreur signée moyenne) mesure une tendance systématique à sur- ou "
            "sous-estimer — ici il est proche de 0, ce qui est un bon signe, mais ce n'est **pas** une "
            "mesure de précision par vol : des erreurs positives et négatives peuvent s'annuler en "
            "moyenne même si chaque prédiction individuelle est imprécise. La précision réelle du "
            "modèle se lit sur le **MAE** (erreur absolue moyenne), cohérent avec la page « Modélisation ML ». "
            "Notez aussi que ce lot mélange les vols utilisés à l'entraînement (80%) et les vols de test "
            "jamais vus par le modèle (20%) — la précision sur des vols réellement nouveaux est mieux "
            "estimée par le MAE de test affiché sur la page « Modélisation ML »."
        )

        st.dataframe(
            batch[
                [
                    "flight_id", "ac_registration", "ac_type", "route",
                    "trip_fuel", "taxi_fuel",
                    "fuel_reel_consomme", "fuel_reel_consomme_predit", "fuel_reel_consomme_ecart",
                    "fuel_score", "recommandation",
                ]
            ].head(300),
            use_container_width=True,
        )
