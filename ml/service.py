"""Prediction service: loads the persisted AFO-optimised pipeline and scores
patient records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, ZERO_AS_MISSING
from .trainer import ARTIFACT_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT = ARTIFACT_PATH


class PredictionService:
    """Thin wrapper around the saved model bundle."""

    def __init__(self, artifact_path: Optional[Path] = None) -> None:
        self.artifact_path = Path(artifact_path or DEFAULT_ARTIFACT)
        self.bundle: Dict = {}
        self.load()

    def load(self) -> None:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found at {}. Run: python train_models.py".format(
                    self.artifact_path
                )
            )
        self.bundle = joblib.load(self.artifact_path)
        required = ["pipelines", "selected_features", "best_model_key", "metrics"]
        missing = [key for key in required if key not in self.bundle]
        if missing:
            raise ValueError("Model artifact is incomplete: {}".format(missing))

    @property
    def metadata(self) -> Dict:
        return {key: value for key, value in self.bundle.items() if key != "pipelines"}

    @property
    def is_ready(self) -> bool:
        return bool(self.bundle)

    def predict(self, values: Dict[str, float]) -> Dict:
        """Run the selected AFO-optimised classifier on one patient record."""
        selected = self.bundle["selected_features"]
        model_key = self.bundle["best_model_key"]
        pipeline = self.bundle["pipelines"]["optimized"][model_key]

        record = {feature: float(values[feature]) for feature in selected}
        frame = pd.DataFrame([record], columns=selected)

        # Match training-time cleaning: zeros in these fields mean "missing".
        imputed_fields = []
        for feature in selected:
            if feature in ZERO_AS_MISSING and frame.at[0, feature] == 0:
                frame.at[0, feature] = np.nan
                imputed_fields.append(feature)

        # NumPy input matches how the pipelines were fitted during training.
        probability = float(pipeline.predict_proba(frame.to_numpy(dtype=float))[0, 1])
        label = int(probability >= self.bundle.get("decision_threshold", 0.5))
        risk_level = self._risk_level(probability)

        return {
            "prediction": label,
            "label": "Diabetic" if label == 1 else "Non-Diabetic",
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100.0, 1),
            "confidence_percent": round(
                max(probability, 1.0 - probability) * 100.0, 1
            ),
            "risk_level": risk_level["name"],
            "risk_class": risk_level["css"],
            "risk_message": risk_level["message"],
            "model_key": model_key,
            "model_name": self.bundle["best_model_name"],
            "model_family": self.bundle["best_model_family"],
            "selected_features": selected,
            "used_features": len(selected),
            "total_features": len(FEATURE_COLUMNS),
            "imputed_fields": imputed_fields,
            "threshold": self.bundle.get("decision_threshold", 0.5),
        }

    @staticmethod
    def _risk_level(probability: float) -> Dict[str, str]:
        if probability >= 0.60:
            return {
                "name": "High",
                "css": "high",
                "message": (
                    "The model places this record in the high-probability band. "
                    "Share the screening output with a qualified clinician for "
                    "confirmatory testing."
                ),
            }
        if probability >= 0.35:
            return {
                "name": "Moderate",
                "css": "moderate",
                "message": (
                    "The model places this record in the moderate-probability "
                    "band. A clinical review and follow-up assessment are advised."
                ),
            }
        return {
            "name": "Low",
            "css": "low",
            "message": (
                "The model places this record in the low-probability band. "
                "Routine monitoring and healthy lifestyle habits remain advisable."
            ),
        }
