"""
Module de centralisation des donnees (Sprint 1).

Lit un export de vols Nouvelair (CSV ";" avec decimales francaises), applique
les regles de nettoyage definies dans le rapport de stage (section 4.3) et
calcule les variables derivees (section 4.4), puis charge le resultat dans
la base SQLite centralisee.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "fuel_platform.db"

# Colonnes du CSV source -> noms de colonnes normalises dans la base
COLUMN_MAP = {
    "Date of operation (UTC)": "date_operation",
    "AC registration": "ac_registration",
    "Flight ID": "flight_id",
    "ICAO Call sign": "icao_call_sign",
    "Flight type": "flight_type",
    "Departing Airport ICAO Code": "dep_icao",
    "Departing Airport": "dep_airport",
    "Destination Airport ICAO Code": "dest_icao",
    "Destination Airport": "dest_airport",
    "Scheduled Departuree": "scheduled_departure",
    "Actual Departuree": "actual_departure",
    "Scheduled Arrivall": "scheduled_arrival",
    "Actual Arrivall": "actual_arrival",
    "Block Off (tonnes)": "block_off",
    "Block On (tonnes)": "block_on",
    "Uplift Volume (Litres)": "uplift_volume",
    "Uplift density": "uplift_density",
    "Adults": "adults",
    "Children": "children",
    "Pax": "pax",
    "PAX Embarqué": "pax_embarque",
    "Trip Fuel (t)": "trip_fuel",
    "Taxi Fuel (t)": "taxi_fuel",
    "Alternate Fuel (t)": "alternate_fuel",
    "Contingency Fuel (t)": "contingency_fuel",
    "Final Reserve Fuel (t)": "final_reserve_fuel",
    "Discretionary Fuel (t)": "discretionary_fuel",
    "Extra Fuel (t)": "extra_fuel",
    "Trajectory": "trajectory",
    "Fuel burn (t)": "fuel_burn",
    "AC Type": "ac_type",
    "Flight duration": "flight_duration_min",
    "Route": "route",
    "Consommation horaire": "consommation_horaire",
}

FRENCH_DECIMAL_COLUMNS = [
    "block_off", "block_on", "uplift_density", "trip_fuel", "taxi_fuel",
    "alternate_fuel", "contingency_fuel", "final_reserve_fuel",
    "discretionary_fuel", "extra_fuel", "fuel_burn", "consommation_horaire",
]

OUTLIER_PERCENTILE = 0.995
FUEL_COLUMNS_FOR_OUTLIERS = ["trip_fuel", "fuel_burn"]


@dataclass
class CleaningReport:
    lignes_source: int
    lignes_trip_fuel_invalide: int
    lignes_outliers_plafonnees: int
    passagers_imputes: int
    lignes_fuel_reel_invalide: int
    lignes_finales: int


def _to_french_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=";",
        encoding="utf-8-sig",
        engine="python",
        dtype=str,
    )
    df = df.rename(columns=COLUMN_MAP)
    keep = [c for c in COLUMN_MAP.values() if c in df.columns]
    return df[keep].copy()


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    n_source = len(df)

    for col in FRENCH_DECIMAL_COLUMNS:
        if col in df.columns:
            df[col] = _to_french_float(df[col])

    for col in ["adults", "children", "pax", "pax_embarque", "flight_duration_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Regle 1 : exclure les vols avec Trip Fuel negatif ou nul
    invalid_mask = df["trip_fuel"].isna() | (df["trip_fuel"] <= 0)
    n_invalid = int(invalid_mask.sum())
    df = df.loc[~invalid_mask].copy()

    # Regle 2 : plafonner les valeurs aberrantes au-dela du 99,5e centile
    n_capped = 0
    for col in FUEL_COLUMNS_FOR_OUTLIERS:
        if col in df.columns:
            cap = df[col].quantile(OUTLIER_PERCENTILE)
            over = df[col] > cap
            n_capped += int(over.sum())
            df.loc[over, col] = cap

    # Regle 3 : imputer les passagers manquants a partir de Adults + Children
    n_imputed = 0
    if "pax" in df.columns:
        missing_pax = df["pax"].isna()
        if "adults" in df.columns and "children" in df.columns:
            fallback = df["adults"].fillna(0) + df["children"].fillna(0)
            n_imputed = int(missing_pax.sum())
            df.loc[missing_pax, "pax"] = fallback[missing_pax]

    # Regle 4 : normaliser les dates (format francais "jeudi 10 juillet 2025")
    if "date_operation" in df.columns:
        df["date_operation_norm"] = _parse_french_dates(df["date_operation"])
        df["annee"] = df["date_operation_norm"].dt.year
        df["mois"] = df["date_operation_norm"].dt.month
        df["jour_semaine"] = df["date_operation_norm"].dt.dayofweek

    # Feature engineering (section 4.4 du rapport)
    if {"actual_departure", "actual_arrival"}.issubset(df.columns):
        dep = pd.to_datetime(df["actual_departure"], format="%d/%m/%Y %H:%M", errors="coerce")
        arr = pd.to_datetime(df["actual_arrival"], format="%d/%m/%Y %H:%M", errors="coerce")
        df["duree_vol_reelle_h"] = (arr - dep).dt.total_seconds() / 3600

    # Duree PREVUE (connue avant le vol, depuis les horaires programmes) : a utiliser
    # comme feature pour la prediction, contrairement a duree_vol_reelle_h / flight_duration_min
    # qui sont des grandeurs realisees (identiques a posteriori, non disponibles pour un vol futur).
    if {"scheduled_departure", "scheduled_arrival"}.issubset(df.columns):
        sched_dep = pd.to_datetime(df["scheduled_departure"], format="%d/%m/%Y %H:%M", errors="coerce")
        sched_arr = pd.to_datetime(df["scheduled_arrival"], format="%d/%m/%Y %H:%M", errors="coerce")
        df["duree_prevue_min"] = (sched_arr - sched_dep).dt.total_seconds() / 60

    n_fuel_reel_invalide = 0
    if {"block_off", "block_on"}.issubset(df.columns):
        df["fuel_reel_consomme"] = df["block_off"] - df["block_on"]
        # Regle 5 : exclure les vols dont le fuel reel consomme est negatif ou nul
        # (Block Off < Block On : incoherence de mesure, physiquement impossible)
        invalid_fuel_reel = df["fuel_reel_consomme"].isna() | (df["fuel_reel_consomme"] <= 0)
        n_fuel_reel_invalide = int(invalid_fuel_reel.sum())
        df = df.loc[~invalid_fuel_reel].copy()

    if {"fuel_reel_consomme", "trip_fuel"}.issubset(df.columns):
        df["ecart_carburant"] = df["fuel_reel_consomme"] - df["trip_fuel"]

    if {"fuel_reel_consomme", "pax"}.issubset(df.columns):
        df["ratio_fuel_pax"] = df["fuel_reel_consomme"] / df["pax"].replace(0, pd.NA)

    report = CleaningReport(
        lignes_source=n_source,
        lignes_trip_fuel_invalide=n_invalid,
        lignes_outliers_plafonnees=n_capped,
        passagers_imputes=n_imputed,
        lignes_fuel_reel_invalide=n_fuel_reel_invalide,
        lignes_finales=len(df),
    )
    return df.reset_index(drop=True), report


def _parse_french_dates(series: pd.Series) -> pd.Series:
    months = {
        "janvier": "January", "février": "February", "mars": "March",
        "avril": "April", "mai": "May", "juin": "June", "juillet": "July",
        "août": "August", "septembre": "September", "octobre": "October",
        "novembre": "November", "décembre": "December",
    }
    days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

    def convert(value: str) -> str:
        text = str(value).strip().lower()
        for d in days:
            text = text.replace(d + " ", "", 1) if text.startswith(d) else text
        for fr, en in months.items():
            text = text.replace(fr, en)
        return text

    converted = series.map(convert)
    return pd.to_datetime(converted, format="%d %B %Y", errors="coerce")


def load_to_sqlite(df: pd.DataFrame, db_path: str | Path = DB_PATH, table_name: str = "flights") -> int:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return count


def run_pipeline(csv_path: str | Path, db_path: str | Path = DB_PATH) -> CleaningReport:
    raw = load_raw_csv(csv_path)
    clean, report = clean_dataset(raw)
    load_to_sqlite(clean, db_path)
    return report


if __name__ == "__main__":
    default_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "SetNouvelair.csv"
    rpt = run_pipeline(default_csv)
    print(rpt)
