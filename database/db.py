"""SQLite storage for prediction history.

The database contains one table, ``predictions``.  Every assessment submitted
through the web interface is stored with its clinical inputs, the model output,
probability, model name and timestamp.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import current_app, g

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE = BASE_DIR / "database" / "diabetes.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    patient_ref TEXT,
    pregnancies REAL NOT NULL,
    glucose REAL NOT NULL,
    blood_pressure REAL NOT NULL,
    skin_thickness REAL NOT NULL,
    insulin REAL NOT NULL,
    bmi REAL NOT NULL,
    dpf REAL NOT NULL,
    age REAL NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    probability REAL NOT NULL,
    model_key TEXT NOT NULL,
    model_name TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    selected_features TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_result
    ON predictions (prediction);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exception=None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    connection = get_db()
    connection.executescript(SCHEMA)
    connection.commit()


def init_app(app) -> None:
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()


def insert_prediction(
    values: Dict[str, float],
    result: Dict,
    patient_ref: str = "",
    created_at: Optional[str] = None,
) -> int:
    connection = get_db()
    cursor = connection.execute(
        """
        INSERT INTO predictions (
            created_at, patient_ref, pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age, prediction, probability,
            model_key, model_name, risk_level, selected_features
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_ref.strip(),
            values["Pregnancies"],
            values["Glucose"],
            values["BloodPressure"],
            values["SkinThickness"],
            values["Insulin"],
            values["BMI"],
            values["DiabetesPedigreeFunction"],
            values["Age"],
            result["prediction"],
            result["probability"],
            result["model_key"],
            result["model_name"],
            result["risk_level"],
            ", ".join(result["selected_features"]),
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def fetch_prediction(prediction_id: int) -> Optional[sqlite3.Row]:
    return get_db().execute(
        "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
    ).fetchone()


def recent_predictions(limit: int = 6) -> List[sqlite3.Row]:
    return get_db().execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def get_stats() -> Dict:
    row = get_db().execute(
        """
        SELECT COUNT(*) AS total,
               COALESCE(SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END), 0) AS diabetic,
               COALESCE(SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END), 0) AS non_diabetic,
               COALESCE(AVG(probability), 0) AS avg_probability
        FROM predictions
        """
    ).fetchone()
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = get_db().execute(
        "SELECT COUNT(*) AS count FROM predictions WHERE date(created_at) = ?",
        (today,),
    ).fetchone()["count"]

    return {
        "total": int(row["total"]),
        "diabetic": int(row["diabetic"]),
        "non_diabetic": int(row["non_diabetic"]),
        "avg_probability": round(float(row["avg_probability"]) * 100.0, 1),
        "today": int(today_count),
    }


def list_predictions(
    page: int = 1,
    per_page: int = 10,
    query: str = "",
    outcome: str = "",
    model: str = "",
    date_from: str = "",
    date_to: str = "",
) -> Tuple[List[sqlite3.Row], int, Dict]:
    """Search + filter + paginate the prediction history."""
    clauses = []
    params: List = []

    if query:
        clauses.append("(patient_ref LIKE ? OR CAST(id AS TEXT) = ?)")
        params.extend(["%{}%".format(query), query])
    if outcome in ("0", "1"):
        clauses.append("prediction = ?")
        params.append(int(outcome))
    if model:
        clauses.append("model_name = ?")
        params.append(model)
    if date_from:
        clauses.append("date(created_at) >= date(?)")
        params.append(date_from)
    if date_to:
        clauses.append("date(created_at) <= date(?)")
        params.append(date_to)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    connection = get_db()
    total = connection.execute(
        "SELECT COUNT(*) AS count FROM predictions" + where, params
    ).fetchone()["count"]

    page = max(1, int(page))
    per_page = max(1, min(int(per_page), 100))
    offset = (page - 1) * per_page
    rows = connection.execute(
        "SELECT * FROM predictions"
        + where
        + " ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [per_page, offset],
    ).fetchall()

    filters = {
        "query": query,
        "outcome": outcome,
        "model": model,
        "date_from": date_from,
        "date_to": date_to,
    }
    return rows, int(total), filters


def model_options() -> List[str]:
    rows = get_db().execute(
        "SELECT DISTINCT model_name FROM predictions ORDER BY model_name"
    ).fetchall()
    return [row["model_name"] for row in rows]
