"""
Abstraction de source de donnees (DataSource).

La plateforme peut alimenter ses traitements (aperçu, entrainement,
prediction automatique) a partir de deux sources :
  - FICHIER  : un nouvel export CSV depose par l'utilisateur, nettoye a la volee
               (ne modifie pas necessairement la base tant qu'il n'est pas charge)
  - DATABASE : la base SQLite deja centralisee (table `flights`), issue
               d'imports precedents

Cette distinction est importante pour la prediction : le mode "Automatique"
fonctionne toujours sur la DATABASE (le referentiel centralise), jamais sur
un fichier ponctuel, afin de rester reproductible et de ne pas dependre d'un
fichier tenu en memoire.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from ingestion import DB_PATH, CleaningReport, clean_dataset, load_raw_csv, load_to_sqlite


class SourceType(str, Enum):
    FICHIER = "fichier"
    DATABASE = "database"


@dataclass
class DataSourceResult:
    source_type: SourceType
    df: pd.DataFrame
    report: CleaningReport | None = None  # uniquement rempli pour SourceType.FICHIER


def database_exists(db_path: Path = DB_PATH) -> bool:
    return db_path.exists()


def database_row_count(db_path: Path = DB_PATH) -> int:
    if not database_exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        try:
            return conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def load_from_database(db_path: Path = DB_PATH) -> DataSourceResult:
    """Lit directement le referentiel centralise, sans reappliquer le nettoyage."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM flights", conn)
    return DataSourceResult(source_type=SourceType.DATABASE, df=df, report=None)


def load_from_file(file, persist_to_db: bool = False, db_path: Path = DB_PATH) -> DataSourceResult:
    """Lit et nettoie un nouveau fichier. Si persist_to_db=True, remplace la base centralisee."""
    raw_df = load_raw_csv(file)
    clean_df, report = clean_dataset(raw_df)
    if persist_to_db:
        load_to_sqlite(clean_df, db_path)
    return DataSourceResult(source_type=SourceType.FICHIER, df=clean_df, report=report)


def load(source_type: SourceType, file=None, persist_to_db: bool = False, db_path: Path = DB_PATH) -> DataSourceResult:
    if source_type == SourceType.DATABASE:
        return load_from_database(db_path)
    if file is None:
        raise ValueError("Un fichier est requis pour la source FICHIER")
    return load_from_file(file, persist_to_db=persist_to_db, db_path=db_path)
