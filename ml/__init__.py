"""Machine learning and Archerfish Optimization package for the
Intelligent Diabetes Classification System.
"""

from .data import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    FEATURE_UNITS,
    TARGET_COLUMN,
    ZERO_AS_MISSING,
    dataset_summary,
    load_dataset,
)
from .afo import ArcherfishFeatureSelector
from .trainer import MODEL_DISPLAY, train_and_evaluate
from .service import PredictionService

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_LABELS",
    "FEATURE_UNITS",
    "TARGET_COLUMN",
    "ZERO_AS_MISSING",
    "dataset_summary",
    "load_dataset",
    "ArcherfishFeatureSelector",
    "MODEL_DISPLAY",
    "train_and_evaluate",
    "PredictionService",
]
