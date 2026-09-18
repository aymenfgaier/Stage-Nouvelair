"""
Point d'entree de la plateforme d'optimisation de la consommation de fuel.
Sert egalement de plan du site (navigation vers toutes les pages).
"""

import streamlit as st

st.set_page_config(page_title="Plateforme Fuel Nouvelair", page_icon="✈️", layout="wide")

st.title("✈️ Plateforme d'optimisation de la consommation de fuel")
st.caption("Stage ingénieur — Nouvelair — Optimisation prédictive du carburant")

st.markdown(
    "Cette plateforme centralise les données de vol, les nettoie et prédit la "
    "consommation réelle de carburant à partir du plan de vol. Cliquez sur une "
    "page ci-dessous pour y accéder directement."
)

st.divider()

PAGES = [
    {
        "path": "pages/1_Import_Centralisation.py",
        "icon": "📥",
        "title": "Import & Centralisation",
        "description": "Sprint 1 — Importer un nouveau fichier de vols (nettoyage automatique) ou relire la base déjà centralisée.",
    },
    {
        "path": "pages/2_Dashboard.py",
        "icon": "📊",
        "title": "Dashboard",
        "description": "KPI sur les vols centralisés : consommation par flotte/route/mois, écarts carburant.",
    },
    {
        "path": "pages/3_Modelisation_ML.py",
        "icon": "🧠",
        "title": "Modélisation ML",
        "description": "Sprint 2 — Entraînement du modèle Fuel réel total, évaluation, SHAP, détection de dérive.",
    },
    {
        "path": "pages/4_Prediction.py",
        "icon": "🔮",
        "title": "Prédiction",
        "description": "Prédire la consommation d'un vol en mode manuel (vol choisi/saisi) ou automatique (toute la base).",
    },
]

cols = st.columns(2)
for i, page in enumerate(PAGES):
    with cols[i % 2]:
        with st.container(border=True):
            st.page_link(page["path"], label=f"**{page['title']}**", icon=page["icon"], use_container_width=True)
            st.caption(page["description"])

