"""Download and validate the public PIMA Indians Diabetes dataset.

The CSV is already shipped in ``dataset/`` so the application works offline.
This script is provided for reproducibility if the file needs to be restored.

Usage:
    python scripts/download_dataset.py
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DESTINATION = Path(__file__).resolve().parent.parent / "dataset" / "pima-indians-diabetes.csv"
EXPECTED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
EXPECTED_ROWS = 768


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading PIMA dataset from {}".format(URL))
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    DESTINATION.write_bytes(payload)

    with DESTINATION.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    header, data = rows[0], rows[1:]
    if header != EXPECTED_COLUMNS:
        raise SystemExit("Unexpected header: {}".format(header))
    if len(data) != EXPECTED_ROWS:
        raise SystemExit("Expected {} rows, found {}".format(EXPECTED_ROWS, len(data)))

    positives = sum(1 for row in data if row[-1] == "1")
    print("Saved {} ({} records, {} diabetic / {} non-diabetic)".format(
        DESTINATION, len(data), positives, len(data) - positives
    ))


if __name__ == "__main__":
    main()
