"""
Page Streamlit - Sprint 1 : Import & Centralisation des donnees.

Cette page gere deux sources de donnees (DataSource) :
  - Nouveau fichier CSV : nettoie un export et le charge dans la base centralisee.
  - Base de donnees deja centralisee : relit directement le referentiel existant,
    sans reimporter de fichier (utile pour verifier/parcourir l'etat actuel).
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from datasource import SourceType, database_exists, database_row_count, load  # noqa: E402
from ingestion import DB_PATH  # noqa: E402

st.set_page_config(page_title="Import & Centralisation", page_icon="📥", layout="wide")
st.title("📥 Import & Centralisation des données")
st.caption("Sprint 1 — Sources de données et centralisation")

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "SetNouvelair.csv"

source_choice = st.radio(
    "Source des données",
    options=["Nouveau fichier CSV", "Base de données déjà centralisée"],
    horizontal=True,
)

if source_choice == "Nouveau fichier CSV":
    st.markdown(
        "Déposez un export de vols (format identique à `SetNouvelair.csv` : "
        "séparateur `;`, décimales françaises) ou utilisez le dataset déjà présent sur disque. "
        "Le nettoyage remplace le contenu de la base centralisée."
    )

    uploaded = st.file_uploader("Fichier de vols (CSV)", type=["csv"])

    use_default = False
    if uploaded is None and DEFAULT_CSV.exists():
        use_default = st.checkbox(f"Utiliser le fichier déjà présent : {DEFAULT_CSV.name}", value=True)

    source_file = uploaded if uploaded is not None else (DEFAULT_CSV if use_default else None)

    if source_file is None:
        st.warning("Sélectionnez un fichier pour lancer la centralisation.")
        st.stop()

    if st.button("Lancer le nettoyage et la centralisation", type="primary"):
        with st.spinner("Lecture et nettoyage du dataset..."):
            result = load(SourceType.FICHIER, file=source_file, persist_to_db=True)
        st.session_state["last_import_result"] = result
        st.session_state["dataset_centralise"] = True

    if "last_import_result" in st.session_state:
        result = st.session_state["last_import_result"]
        report = result.report
        clean_df = result.df

        st.success(f"{len(clean_df)} vols centralisés dans la base ({DB_PATH.name}).")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Vols source", report.lignes_source)
        c2.metric("Trip Fuel invalide exclus", report.lignes_trip_fuel_invalide)
        c3.metric("Fuel réel invalide exclus", report.lignes_fuel_reel_invalide)
        c4.metric("Outliers plafonnés (>99,5e pct)", report.lignes_outliers_plafonnees)
        c5.metric("Passagers imputés", report.passagers_imputes)
        c6.metric("Vols finaux", report.lignes_finales)

        st.subheader("Aperçu du dataset centralisé")
        st.dataframe(
            clean_df[
                [
                    "date_operation", "ac_registration", "ac_type", "route",
                    "trip_fuel", "taxi_fuel", "fuel_reel_consomme", "ecart_carburant",
                    "duree_prevue_min", "pax",
                ]
            ],
            use_container_width=True,
        )

else:  # Base de données déjà centralisée
    st.markdown(
        "Cette source relit directement le référentiel déjà centralisé — aucun fichier n'est "
        "réimporté. C'est cette même base que les modules d'entraînement et de prédiction "
        "automatique utilisent."
    )

    if not database_exists():
        st.warning("Aucune base centralisée pour l'instant. Passez par « Nouveau fichier CSV » d'abord.")
        st.stop()

    n_rows = database_row_count()
    st.metric("Vols actuellement centralisés", f"{n_rows:,}".replace(",", " "))

    st.button("🔄 Rafraîchir depuis la base", help="La table ci-dessous se charge automatiquement ; utilisez ce bouton après un nouvel import.")

    with st.spinner("Lecture de la base centralisée..."):
        result = load(SourceType.DATABASE)
    st.session_state["dataset_centralise"] = True
    st.subheader("Aperçu du référentiel centralisé")
    st.dataframe(
        result.df[
            [
                "date_operation", "ac_registration", "ac_type", "route",
                "trip_fuel", "taxi_fuel", "fuel_reel_consomme", "ecart_carburant",
                "duree_prevue_min", "pax",
            ]
        ],
        use_container_width=True,
    )
