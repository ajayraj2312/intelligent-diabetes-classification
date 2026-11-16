"""Dataset access and cleaning for the PIMA Indians Diabetes dataset.

The CSV shipped in ``dataset/`` is the canonical UCI / PIMA Indians Diabetes
dataset (768 female patients of Pima Indian heritage, 8 clinical attributes,
binary Outcome).  It is distributed by the UCI Machine Learning Repository and
mirrored by the Plotly public datasets collection.

Preprocessing policy
--------------------
* ``Glucose``, ``BloodPressure``, ``SkinThickness``, ``Insulin`` and ``BMI``
  use the value 0 as a missing-value marker in this dataset.  A zero is not
  physiologically plausible for any of them, so zeros are converted to NaN and
  later imputed with the **training-set median** inside the model pipeline.
* ``Pregnancies`` may legitimately be 0 and is left untouched.
* ``DiabetesPedigreeFunction`` is a unitless score and is left untouched.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "pima-indians-diabetes.csv"

TARGET_COLUMN = "Outcome"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Plasma Glucose",
    "BloodPressure": "Diastolic Blood Pressure",
    "SkinThickness": "Triceps Skin Thickness",
    "Insulin": "Serum Insulin",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age",
}

FEATURE_UNITS = {
    "Pregnancies": "count",
    "Glucose": "mg/dL",
    "BloodPressure": "mm Hg",
    "SkinThickness": "mm",
    "Insulin": "µU/mL",
    "BMI": "kg/m²",
    "DiabetesPedigreeFunction": "score",
    "Age": "years",
}

FEATURE_SHORT = {
    "Pregnancies": "Preg",
    "Glucose": "Glucose",
    "BloodPressure": "BP",
    "SkinThickness": "Skin",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "DPF",
    "Age": "Age",
}

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw PIMA dataset from disk (no cleaning applied)."""
    csv_path = Path(path) if path else DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            "PIMA dataset not found at {}. Place pima-indians-diabetes.csv in "
            "the dataset/ directory or run scripts/download_dataset.py.".format(csv_path)
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: {}".format(missing))
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a modelling copy with implausible zeros converted to NaN."""
    cleaned = df.copy()
    for column in ZERO_AS_MISSING:
        cleaned.loc[cleaned[column] == 0, column] = np.nan
    return cleaned


def dataset_summary(df: pd.DataFrame, csv_path: Optional[Path] = None) -> Dict:
    """Build the dataset facts displayed on the dashboard and AFO pages."""
    cleaned = clean_dataset(df)
    counts = df[TARGET_COLUMN].value_counts().to_dict()
    missing = {
        column: int((df[column] == 0).sum()) for column in ZERO_AS_MISSING
    }
    path = Path(csv_path) if csv_path else DATASET_PATH
    digest = ""
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    return {
        "rows": int(df.shape[0]),
        "feature_count": len(FEATURE_COLUMNS),
        "diabetic": int(counts.get(1, 0)),
        "non_diabetic": int(counts.get(0, 0)),
        "missing_by_feature": missing,
        "missing_total": int(sum(missing.values())),
        "source": "PIMA Indians Diabetes Dataset (UCI)",
        "file": path.name,
        "sha256_short": digest,
        "means": {
            column: round(float(cleaned[column].mean()), 3)
            for column in FEATURE_COLUMNS
        },
    }
