"""
Page Streamlit - Sprint 2 : Modelisation ML, evaluation, SHAP et detection d'anomalies.

Un seul modele : Fuel reel total (validation plan vs reel).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2] / "ml"))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from ingestion import DB_PATH  # noqa: E402
from model_training import MODEL_SPECS, RANDOM_STATE, TARGETS, TEST_SIZE, XGB_HYPERPARAMS, compute_shap_importance, load_flights, train_and_evaluate_all, train_anomaly_detection  # noqa: E402

st.set_page_config(page_title="Modélisation ML", page_icon="🧠", layout="wide")
st.title("🧠 Modélisation ML — Fuel réel total")
st.caption("Sprint 2 — Entraînement, évaluation face à la baseline, explicabilité SHAP et détection de dérive")

if not DB_PATH.exists():
    st.warning("Aucune donnée centralisée. Passez d'abord par la page « Import & Centralisation ».")
    st.stop()

df_all = load_flights()
df_all["date_operation_norm"] = pd.to_datetime(df_all["date_operation_norm"], errors="coerce")
date_min = df_all["date_operation_norm"].min().date()
date_max = df_all["date_operation_norm"].max().date()

periode = st.date_input(
    "📅 Période d'entraînement",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
    help="Le modèle est entraîné et évalué (split 80/20) uniquement sur les vols de cette période.",
)
if isinstance(periode, tuple) and len(periode) == 2:
    debut, fin = periode
else:
    debut, fin = date_min, date_max

df = df_all[(df_all["date_operation_norm"].dt.date >= debut) & (df_all["date_operation_norm"].dt.date <= fin)]
st.caption(f"{len(df):,}".replace(",", " ") + f" vols dans la période sélectionnée (sur {len(df_all):,}".replace(",", " ") + " au total).")

with st.expander("⚙️ Paramètres d'entraînement (hyperparamètres du modèle)", expanded=False):
    params_df = pd.DataFrame(
        {
            "Paramètre": [
                "Répartition train / test", "Graine aléatoire (random_state)",
                "Nombre d'arbres (n_estimators)", "Profondeur maximale (max_depth)",
                "Taux d'apprentissage (learning_rate)", "Sous-échantillonnage lignes (subsample)",
                "Sous-échantillonnage colonnes (colsample_bytree)",
            ],
            "Valeur": [
                f"{int((1 - TEST_SIZE) * 100)}% / {int(TEST_SIZE * 100)}%", RANDOM_STATE,
                XGB_HYPERPARAMS["n_estimators"], XGB_HYPERPARAMS["max_depth"],
                XGB_HYPERPARAMS["learning_rate"], XGB_HYPERPARAMS["subsample"],
                XGB_HYPERPARAMS["colsample_bytree"],
            ],
        }
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)

if df.empty:
    st.warning("Aucun vol dans la période sélectionnée.")
    st.stop()

if st.button("Entraîner le modèle et évaluer", type="primary"):
    with st.spinner("Entraînement sur la période sélectionnée..."):
        st.session_state["ml_results"] = train_and_evaluate_all(df)
    with st.spinner("Détection des dérives par immatriculation (Isolation Forest)..."):
        st.session_state["anomalies"] = train_anomaly_detection()
    with st.spinner("Calcul de l'explicabilité SHAP..."):
        st.session_state["shap_importance"] = {t: compute_shap_importance(t, df=df) for t in TARGETS}

if "ml_results" not in st.session_state:
    st.stop()

results = st.session_state["ml_results"]

for target in TARGETS:
    r = results[target]
    label = MODEL_SPECS[target]["label"]
    st.subheader(label)
    st.caption(f"Entraînement : {r['n_train']} vols (80%) — Test : {r['n_test']} vols (20%), split aléatoire")

    metrics_df = pd.DataFrame(
        {
            "Modèle": ["Baseline (moyenne historique par route)", "Régression linéaire", "XGBoost (Gradient Boosting)"],
            "MAE (t)": [r["baseline"]["mae"], r["linear_regression"]["mae"], r["xgboost"]["mae"]],
            "RMSE (t)": [r["baseline"]["rmse"], r["linear_regression"]["rmse"], r["xgboost"]["rmse"]],
            "R²": [r["baseline"]["r2"], r["linear_regression"]["r2"], r["xgboost"]["r2"]],
            "MAPE (%)": [r["baseline"]["mape"] * 100, r["linear_regression"]["mape"] * 100, r["xgboost"]["mape"] * 100],
        }
    )
    st.dataframe(
        metrics_df.style.format({"MAE (t)": "{:.3f}", "RMSE (t)": "{:.3f}", "R²": "{:.3f}", "MAPE (%)": "{:.2f}"}),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(f"MAE XGBoost — {label}", f"{r['xgboost']['mae']:.3f} t")
    c2.metric("R² XGBoost", f"{r['xgboost']['r2']:.3f}")
    c3.metric("MAPE XGBoost", f"{r['xgboost']['mape']*100:.2f} %")

    if "shap_importance" in st.session_state:
        shap_df = st.session_state["shap_importance"][target].copy()
        shap_df["feature"] = shap_df["feature"].str.replace("remainder__", "", regex=False).str.replace("cat__", "", regex=False)
        st.caption(f"Importance des variables (SHAP) — {label} — top 15 (aéroports individuels regroupés au-delà)")
        st.bar_chart(shap_df.head(15).set_index("feature")["mean_abs_shap"])

    st.divider()

st.subheader("Détection de dérive par immatriculation (Isolation Forest)")
if "anomalies" in st.session_state:
    anomalies_df = st.session_state["anomalies"]
    n_alerts = int(anomalies_df["alerte"].sum())
    st.metric("Avions en alerte de dérive", n_alerts)
    st.dataframe(
        anomalies_df.style.format({"score_derive": "{:.3f}", "taux_alerte_recent": "{:.1%}"}),
        use_container_width=True,
    )
