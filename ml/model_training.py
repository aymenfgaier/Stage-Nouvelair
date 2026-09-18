"""
Module de modelisation ML (Sprint 2).

Deux variantes du meme modele sont entrainees, predisant toutes les deux le
Fuel reel total (Block Off - Block On) :

  - fuel_reel_consomme         : utilise Trip Fuel/Taxi Fuel PLANIFIES comme features
                                  en plus des caracteristiques du vol. Plus precise
                                  (R2 0.963). Utilisee partout SAUF en saisie manuelle :
                                  page Modelisation ML (entrainement/evaluation/SHAP),
                                  mode Automatique, et mode Manuel "vol existant" (le
                                  plan reel est alors connu, tire de l'historique).

  - fuel_reel_consomme_manuel  : n'utilise QUE des caracteristiques du vol connues avant
                                  meme le depot du plan de vol (pas de Trip Fuel/Taxi
                                  Fuel). Legerement moins precise (R2 0.945), mais ne
                                  necessite aucune saisie de plan de vol -- utilisee
                                  uniquement pour la saisie manuelle d'un vol hypothetique
                                  (page Prediction, mode Manuel "saisir manuellement").

Fuite de donnees evitee deliberement : Block Off, Block On, Alternate/Contingency/Final
Reserve Fuel sont EXCLUS des features des deux modeles. Verification empirique : Block
Off = Trip Fuel + Taxi Fuel + reserves (corr. 0.999, ecart moyen 0.024 t) -- ce sont des
grandeurs calculees A PARTIR de trip_fuel/taxi_fuel, pas des mesures independantes.

`flight_duration_min` (duree REALISEE, identique a duree_vol_reelle_h - correlation 1.0)
est egalement exclu au profit de `duree_prevue_min` (duree PROGRAMMEE, connue avant le vol).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "fuel_platform.db"
MODELS_DIR = Path(__file__).resolve().parent / "models"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyperparametres du modele XGBoost retenu -- centralises ici pour etre a la fois
# utilises a l'entrainement et affiches tels quels sur la page Modelisation ML.
XGB_HYPERPARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": RANDOM_STATE,
}

# Specification de chaque variante : colonne cible reelle + variables numeriques/categorielles.
MODEL_SPECS: dict[str, dict] = {
    "fuel_reel_consomme": {
        "target_column": "fuel_reel_consomme",
        "label": "Fuel réel total",
        "numeric": ["adults", "children", "pax", "duree_prevue_min", "mois", "jour_semaine", "trip_fuel", "taxi_fuel"],
        "categorical": ["ac_type", "dep_icao", "dest_icao"],
    },
    "fuel_reel_consomme_manuel": {
        "target_column": "fuel_reel_consomme",
        "label": "Fuel réel total (saisie manuelle, sans plan de vol)",
        "numeric": ["adults", "children", "pax", "duree_prevue_min", "mois", "jour_semaine"],
        "categorical": ["ac_type", "dep_icao", "dest_icao"],
    },
}
# Seul le modele principal est entraine/affiche sur la page Modelisation ML et utilise
# en mode Automatique -- le modele "_manuel" est entraine en meme temps (train_and_evaluate_all)
# mais reserve a la saisie manuelle sur la page Prediction.
TARGETS = ["fuel_reel_consomme"]


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float

    def as_dict(self) -> dict:
        return asdict(self)


def load_flights(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM flights", conn)
    required = [
        "taxi_fuel", "trip_fuel", "fuel_reel_consomme", "ac_type", "dep_icao", "dest_icao",
        "adults", "children", "duree_prevue_min", "pax",
    ]
    df = df.dropna(subset=required).copy()
    df = df[(df["taxi_fuel"] > 0) & (df["trip_fuel"] > 0) & (df["fuel_reel_consomme"] > 0)]
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2=float(r2_score(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred)),
    )


def _route_baseline_col(target: str) -> str:
    return f"route_baseline_{target}"


def _airport_pair_key(dep_icao: str, dest_icao: str) -> str:
    return f"{dep_icao}|{dest_icao}"


def add_route_baseline(train_df: pd.DataFrame, other_df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Moyenne historique de la cible par PAIRE D'AEROPORTS (dep_icao, dest_icao),
    calculee uniquement sur le train, avec repli sur la moyenne globale."""
    target_column = MODEL_SPECS[target]["target_column"]
    col = _route_baseline_col(target)
    train_df = train_df.copy()
    other_df = other_df.copy()

    train_df["_pair_key"] = [
        _airport_pair_key(d, a) for d, a in zip(train_df["dep_icao"], train_df["dest_icao"])
    ]
    pair_means = train_df.groupby("_pair_key")[target_column].mean()
    global_mean = train_df[target_column].mean()

    other_df["_pair_key"] = [
        _airport_pair_key(d, a) for d, a in zip(other_df["dep_icao"], other_df["dest_icao"])
    ]
    train_df[col] = train_df["_pair_key"].map(pair_means)
    other_df[col] = other_df["_pair_key"].map(pair_means).fillna(global_mean)

    train_df.drop(columns="_pair_key", inplace=True)
    other_df.drop(columns="_pair_key", inplace=True)
    return train_df, other_df, {"pair_means": pair_means.to_dict(), "global_mean": float(global_mean)}


def get_feature_columns(target: str) -> list[str]:
    spec = MODEL_SPECS[target]
    return spec["categorical"] + spec["numeric"] + [_route_baseline_col(target)]


def build_pipeline(target: str, model) -> Pipeline:
    categorical = MODEL_SPECS[target]["categorical"]
    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train_and_evaluate_target(df: pd.DataFrame, target: str) -> dict:
    target_column = MODEL_SPECS[target]["target_column"]
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df, test_df, route_info = add_route_baseline(train_df, test_df, target)

    feature_cols = get_feature_columns(target)
    X_train, y_train = train_df[feature_cols], train_df[target_column]
    X_test, y_test = test_df[feature_cols], test_df[target_column]

    baseline_pred = test_df[_route_baseline_col(target)]
    baseline_metrics = compute_metrics(y_test, baseline_pred)

    lin_pipeline = build_pipeline(target, LinearRegression())
    lin_pipeline.fit(X_train, y_train)
    lin_metrics = compute_metrics(y_test, lin_pipeline.predict(X_test))

    xgb_pipeline = build_pipeline(
        target,
        XGBRegressor(**XGB_HYPERPARAMS),
    )
    xgb_pipeline.fit(X_train, y_train)
    xgb_metrics = compute_metrics(y_test, xgb_pipeline.predict(X_test))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lin_pipeline, MODELS_DIR / f"{target}_linear_regression.joblib")
    joblib.dump(xgb_pipeline, MODELS_DIR / f"{target}_xgboost.joblib")
    joblib.dump(route_info, MODELS_DIR / f"{target}_route_baseline.joblib")

    result = {
        "target": target,
        "label": MODEL_SPECS[target]["label"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "baseline": baseline_metrics.as_dict(),
        "linear_regression": lin_metrics.as_dict(),
        "xgboost": xgb_metrics.as_dict(),
        "feature_columns": feature_cols,
    }

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS models_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT, target TEXT, trained_at TEXT,
                n_train INTEGER, n_test INTEGER, metrics_json TEXT
            )"""
        )
        for name, metrics in [("baseline", baseline_metrics), ("linear_regression", lin_metrics), ("xgboost", xgb_metrics)]:
            conn.execute(
                "INSERT INTO models_metadata (model_name, target, trained_at, n_train, n_test, metrics_json) VALUES (?,?,datetime('now'),?,?,?)",
                (name, target, len(train_df), len(test_df), json.dumps(metrics.as_dict())),
            )
        conn.commit()

    return result


def train_and_evaluate_all(df: pd.DataFrame | None = None) -> dict:
    """Entraine TOUTES les variantes de MODEL_SPECS (y compris le modele reserve a la
    saisie manuelle), mais seul TARGETS est retourne/affiche par la page Modelisation ML."""
    if df is None:
        df = load_flights()
    all_results = {key: train_and_evaluate_target(df, key) for key in MODEL_SPECS}
    return {target: all_results[target] for target in TARGETS}


def compute_shap_importance(target: str, df: pd.DataFrame | None = None, sample_size: int = 2000) -> pd.DataFrame:
    import shap

    if df is None:
        df = load_flights()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df, test_df, _ = add_route_baseline(train_df, test_df, target)
    feature_cols = get_feature_columns(target)

    xgb_pipeline: Pipeline = joblib.load(MODELS_DIR / f"{target}_xgboost.joblib")
    preprocess = xgb_pipeline.named_steps["preprocess"]
    model = xgb_pipeline.named_steps["model"]

    sample = test_df[feature_cols].sample(min(sample_size, len(test_df)), random_state=RANDOM_STATE)
    X_transformed = preprocess.transform(sample)
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance


def train_anomaly_detection(min_flights: int = 15, contamination: float = 0.08) -> pd.DataFrame:
    """Isolation Forest par immatriculation sur l'ecart carburant et la consommation horaire."""
    df = load_flights()
    results = []

    for reg, group in df.groupby("ac_registration"):
        if len(group) < min_flights:
            continue
        features = group[["ecart_carburant", "consommation_horaire"]].dropna()
        if len(features) < min_flights:
            continue

        clf = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
        clf.fit(features)
        scores = -clf.decision_function(features)
        flags = clf.predict(features) == -1

        recent_n = max(5, len(features) // 5)
        recent_drift_score = float(scores[-recent_n:].mean())
        recent_alert_rate = float(flags[-recent_n:].mean())

        results.append(
            {
                "ac_registration": reg,
                "n_vols": len(features),
                "score_derive": recent_drift_score,
                "taux_alerte_recent": recent_alert_rate,
                "alerte": recent_alert_rate > 0.15,
            }
        )

    result_df = pd.DataFrame(results).sort_values("score_derive", ascending=False).reset_index(drop=True)
    with sqlite3.connect(DB_PATH) as conn:
        result_df.to_sql("anomaly_scores", conn, if_exists="replace", index=False)
    return result_df


# --------------------------------------------------------------------------
# Prediction (mode Manuel et mode Automatique)
# --------------------------------------------------------------------------

def _load_target_models(target: str):
    model_path = MODELS_DIR / f"{target}_xgboost.joblib"
    baseline_path = MODELS_DIR / f"{target}_route_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle '{target}' introuvable — lancez d'abord l'entraînement.")
    pipeline = joblib.load(model_path)
    route_info = joblib.load(baseline_path)
    return pipeline, route_info


def _resolve_pair_baseline(dep_icao: str, dest_icao: str, route_info: dict) -> float:
    key = _airport_pair_key(dep_icao, dest_icao)
    return route_info["pair_means"].get(key, route_info["global_mean"])


def predict_manual_with_plan(
    ac_type: str, dep_icao: str, dest_icao: str,
    adults: int, children: int, pax: int, duree_prevue_min: float, mois: int, jour_semaine: int,
    trip_fuel: float, taxi_fuel: float,
) -> float:
    """Predit le fuel reel total a partir des caracteristiques du vol ET du plan (Trip
    Fuel/Taxi Fuel) -- utilise pour le mode Manuel "vol existant" (plan connu par l'historique)."""
    target = "fuel_reel_consomme"
    pipeline, route_info = _load_target_models(target)
    row = pd.DataFrame(
        [
            {
                "ac_type": ac_type,
                "dep_icao": dep_icao,
                "dest_icao": dest_icao,
                "adults": adults,
                "children": children,
                "pax": pax,
                "duree_prevue_min": duree_prevue_min,
                "mois": mois,
                "jour_semaine": jour_semaine,
                "trip_fuel": trip_fuel,
                "taxi_fuel": taxi_fuel,
                _route_baseline_col(target): _resolve_pair_baseline(dep_icao, dest_icao, route_info),
            }
        ]
    )
    return float(pipeline.predict(row)[0])


def predict_manual_without_plan(
    ac_type: str, dep_icao: str, dest_icao: str,
    adults: int, children: int, pax: int, duree_prevue_min: float, mois: int, jour_semaine: int,
) -> float:
    """Predit le fuel reel total a partir des SEULES caracteristiques du vol, sans plan de
    vol -- utilise pour le mode Manuel "saisir manuellement" (vol hypothetique)."""
    target = "fuel_reel_consomme_manuel"
    pipeline, route_info = _load_target_models(target)
    row = pd.DataFrame(
        [
            {
                "ac_type": ac_type,
                "dep_icao": dep_icao,
                "dest_icao": dest_icao,
                "adults": adults,
                "children": children,
                "pax": pax,
                "duree_prevue_min": duree_prevue_min,
                "mois": mois,
                "jour_semaine": jour_semaine,
                _route_baseline_col(target): _resolve_pair_baseline(dep_icao, dest_icao, route_info),
            }
        ]
    )
    return float(pipeline.predict(row)[0])


def predict_batch_from_database(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Mode Automatique : score l'ensemble des vols de la base centralisee (jamais un
    fichier), avec le modele principal (avec plan de vol). Calcule egalement le Fuel
    Score et une recommandation carburant par vol (module `recommendation`)."""
    from recommendation import ECART_SIGNIFICATIF_T, FUEL_SCORE_SCALE

    df = load_flights(db_path)
    output = df[
        ["flight_id", "ac_registration", "ac_type", "route", "date_operation_norm", "taxi_fuel", "trip_fuel", "fuel_reel_consomme"]
    ].copy()

    for target in TARGETS:
        target_column = MODEL_SPECS[target]["target_column"]
        pipeline, route_info = _load_target_models(target)
        features = df.copy()
        pair_keys = [_airport_pair_key(d, a) for d, a in zip(features["dep_icao"], features["dest_icao"])]
        features[_route_baseline_col(target)] = [
            route_info["pair_means"].get(k, route_info["global_mean"]) for k in pair_keys
        ]
        X = features[get_feature_columns(target)]
        output[f"{target_column}_predit"] = pipeline.predict(X)
        output[f"{target_column}_ecart"] = output[f"{target_column}_predit"] - output[target_column]

    # Fuel Score (0-100) : ecart entre reel et predit, vectorise (memes regles que
    # recommendation.compute_fuel_score, applique par vol -- pas de donnee pilote disponible).
    predicted = output["fuel_reel_consomme_predit"]
    ecart_pct = (output["fuel_reel_consomme"] - predicted) / predicted.replace(0, pd.NA)
    output["fuel_score"] = (50.0 - ecart_pct * 100.0 * FUEL_SCORE_SCALE).clip(0, 100).fillna(50.0)

    # Recommandation carburant : ecart predit vs plan (Trip Fuel + Taxi Fuel)
    fuel_planifie = output["trip_fuel"] + output["taxi_fuel"]
    ecart_vs_plan = predicted - fuel_planifie
    output["fuel_planifie"] = fuel_planifie
    output["ecart_vs_plan"] = ecart_vs_plan
    output["fuel_optimal_propose"] = predicted * 1.02
    output["recommandation"] = pd.cut(
        ecart_vs_plan,
        bins=[-float("inf"), -ECART_SIGNIFICATIF_T, ECART_SIGNIFICATIF_T, float("inf")],
        labels=["Réduction du carburant discrétionnaire envisageable", "Conforme au plan actuel", "Complément de carburant discrétionnaire recommandé"],
    ).astype(str)

    with sqlite3.connect(db_path) as conn:
        output.to_sql("predictions", conn, if_exists="replace", index=False)

    return output


if __name__ == "__main__":
    data = load_flights()
    print("Vols exploitables pour la modelisation :", len(data))
    results = train_and_evaluate_all(data)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    anomalies = train_anomaly_detection()
    print(anomalies.head(10))
    batch = predict_batch_from_database()
    print(batch.head(10))
