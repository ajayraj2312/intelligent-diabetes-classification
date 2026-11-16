"""Train the AFO feature selector and all eight classification pipelines.

Usage:
    python train_models.py

Writes:
    models/artifacts.joblib   full pipelines + metrics + AFO trace
    models/metadata.json      human-readable metadata (no pickled objects)

The split, scaling, imputation and AFO seed are fixed, so results are
reproducible on every run.
"""

from ml.trainer import train_and_evaluate


if __name__ == "__main__":
    print("=" * 72)
    print("Intelligent Diabetes Classification — AFO + ML training pipeline")
    print("=" * 72)
    bundle = train_and_evaluate()
    print("\nTraining complete.")
    print("Features selected by AFO: {}".format(", ".join(bundle["selected_features"])))
    print("Selected model: {}".format(bundle["best_model_name"]))
    print(
        "Test accuracy: {:.2%}  |  F1: {:.2%}  |  AUC: {:.4f}".format(
            bundle["best_metrics"]["accuracy"],
            bundle["best_metrics"]["f1"],
            bundle["best_metrics"]["roc_auc"],
        )
    )
