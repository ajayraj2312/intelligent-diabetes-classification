"""Reproducible training pipeline for the diabetes classification system.

Pipeline
--------
1. Load the PIMA dataset and convert implausible zeros to missing values.
2. Stratified 80/20 train-test split (``random_state=42``).
3. Run Archerfish Optimization on the training split to select features.
4. Train SVM, Random Forest, KNN and Gaussian Naive Bayes twice:
   on all 8 features (conventional) and on the AFO subset (proposed).
5. Evaluate every model on the untouched test split and persist the pipelines,
   metrics, AFO trace and dataset facts in ``models/artifacts.joblib`` plus a
   human-readable ``models/metadata.json``.

No metric is hard-coded: everything in the UI is read from the saved run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .afo import ArcherfishFeatureSelector
from .data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_dataset,
    dataset_summary,
    load_dataset,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACT_PATH = MODELS_DIR / "artifacts.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

RANDOM_STATE = 42

MODEL_DISPLAY = {
    "svm": "Support Vector Machine",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
}

MODEL_FAMILY = {
    "svm": "Kernel-based classifier (RBF)",
    "random_forest": "Ensemble of decision trees",
    "knn": "Instance-based classifier",
    "naive_bayes": "Probabilistic generative model",
}


def make_estimator(name: str):
    """Return an untrained estimator with documented, fixed hyperparameters."""
    if name == "svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    if name == "naive_bayes":
        return GaussianNB()
    raise ValueError("Unknown model: {}".format(name))


def make_pipeline(name: str) -> Pipeline:
    """Imputation -> scaling -> classifier, fitted inside cross-validation."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", make_estimator(name)),
        ]
    )


def _roc_points(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, List[float]]:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    if len(fpr) > 40:
        indices = np.unique(np.linspace(0, len(fpr) - 1, 40).astype(int))
        fpr, tpr = fpr[indices], tpr[indices]
    return {
        "fpr": [round(float(v), 4) for v in fpr],
        "tpr": [round(float(v), 4) for v in tpr],
    }


def evaluate_pipeline(
    pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> Dict:
    """Calculate all reported metrics for a fitted pipeline on the test split."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    roc_auc = float(roc_auc_score(y_test, probabilities))
    accuracy = float(accuracy_score(y_test, predictions))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "sensitivity": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "test_samples": int(len(y_test)),
        "roc": _roc_points(y_test, probabilities),
        "probabilities": [round(float(p), 4) for p in probabilities],
        "predictions": [int(p) for p in predictions],
    }


def _cross_validation(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        clone(pipeline), X_train, y_train, cv=folds, scoring="accuracy", n_jobs=1
    )
    return {
        "cv_accuracy": round(float(scores.mean()), 4),
        "cv_std": round(float(scores.std()), 4),
        "folds": 5,
    }


def train_and_evaluate(
    dataset_path=None,
    n_archers: int = 22,
    n_iterations: int = 12,
    alpha: float = 0.02,
    verbose: bool = True,
) -> Dict:
    """Run the full training pipeline, persist artifacts and return the bundle."""
    started = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(dataset_path)
    summary = dataset_summary(raw_df, dataset_path)
    df = clean_dataset(raw_df)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    if verbose:
        print("Dataset: {} records, {} features".format(*X.shape))
        print(
            "Split: {} train / {} test".format(len(X_train), len(X_test)),
        )
        print(
            "Running Archerfish Optimization ({} archers, {} iterations)...".format(
                n_archers, n_iterations
            )
        )

    selector = ArcherfishFeatureSelector(
        n_archers=n_archers,
        n_iterations=n_iterations,
        alpha=alpha,
        cv_folds=3,
        surrogate="forest",
        random_state=RANDOM_STATE,
        verbose=verbose,
    )
    afo = selector.fit(X_train, y_train, FEATURE_COLUMNS)
    selected_features = afo["selected_features"]
    selected_idx = [FEATURE_COLUMNS.index(f) for f in selected_features]

    if verbose:
        print("AFO selected {}: {}".format(len(selected_features), selected_features))
        print("AFO fitness: {:.4f} (baseline {:.4f})".format(
            afo["best_fitness"], afo["baseline_fitness"]
        ))

    configs: Dict[str, Dict] = {
        "baseline": {"columns": list(range(len(FEATURE_COLUMNS))), "features": list(FEATURE_COLUMNS)},
        "optimized": {"columns": selected_idx, "features": selected_features},
    }

    pipelines: Dict[str, Dict[str, Pipeline]] = {"baseline": {}, "optimized": {}}
    metrics: Dict[str, Dict[str, Dict]] = {"baseline": {}, "optimized": {}}

    for config_name, config in configs.items():
        columns = config["columns"]
        for model_key in MODEL_DISPLAY:
            pipeline = make_pipeline(model_key)
            pipeline.fit(X_train[:, columns], y_train)
            evaluation = evaluate_pipeline(pipeline, X_test[:, columns], y_test)
            cv = _cross_validation(pipeline, X_train[:, columns], y_train)
            evaluation.update(cv)

            pipelines[config_name][model_key] = pipeline
            metrics[config_name][model_key] = evaluation
            if verbose:
                print(
                    "  {:<10} {:<22} acc={:.4f} precision={:.4f} recall={:.4f} "
                    "f1={:.4f} auc={:.4f}".format(
                        config_name,
                        MODEL_DISPLAY[model_key],
                        evaluation["accuracy"],
                        evaluation["precision"],
                        evaluation["recall"],
                        evaluation["f1"],
                        evaluation["roc_auc"],
                    )
                )

    # Best classifier is chosen on the AFO-optimised feature set (the proposed
    # system), ranked by accuracy then F1 then AUC on the held-out test set.
    best_key = max(
        MODEL_DISPLAY,
        key=lambda key: (
            metrics["optimized"][key]["accuracy"],
            metrics["optimized"][key]["f1"],
            metrics["optimized"][key]["roc_auc"],
        ),
    )
    best_baseline_key = max(
        MODEL_DISPLAY,
        key=lambda key: (
            metrics["baseline"][key]["accuracy"],
            metrics["baseline"][key]["f1"],
            metrics["baseline"][key]["roc_auc"],
        ),
    )

    best = metrics["optimized"][best_key]
    best_baseline = metrics["baseline"][best_baseline_key]

    bundle = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": summary,
        "feature_names": list(FEATURE_COLUMNS),
        "selected_features": selected_features,
        "excluded_features": afo["excluded_features"],
        "afo": afo,
        "pipelines": pipelines,
        "metrics": metrics,
        "best_model_key": best_key,
        "best_model_name": MODEL_DISPLAY[best_key],
        "best_model_family": MODEL_FAMILY[best_key],
        "best_baseline_key": best_baseline_key,
        "best_baseline_name": MODEL_DISPLAY[best_baseline_key],
        "best_metrics": best,
        "best_baseline_metrics": best_baseline,
        "accuracy_gain": round(best["accuracy"] - best_baseline["accuracy"], 4),
        "f1_gain": round(best["f1"] - best_baseline["f1"], 4),
        "auc_gain": round(best["roc_auc"] - best_baseline["roc_auc"], 4),
        "split": {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "test_percent": 20,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "model_display": MODEL_DISPLAY,
        "model_family": MODEL_FAMILY,
        "decision_threshold": 0.5,
        "training_runtime_seconds": round(time.time() - started, 2),
    }

    joblib.dump(bundle, ARTIFACT_PATH)

    metadata = {key: value for key, value in bundle.items() if key != "pipelines"}
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if verbose:
        print(
            "\nSelected model: {} (AFO-optimized)".format(MODEL_DISPLAY[best_key])
        )
        print(
            "Best AFO model accuracy {:.2%} vs best conventional {:.2%} "
            "({:+.2f} pp)".format(
                best["accuracy"], best_baseline["accuracy"],
                100.0 * bundle["accuracy_gain"],
            )
        )
        print("Artifacts saved to {}".format(ARTIFACT_PATH))
        print("Total training time: {:.1f}s".format(bundle["training_runtime_seconds"]))

    return bundle


if __name__ == "__main__":
    train_and_evaluate()
