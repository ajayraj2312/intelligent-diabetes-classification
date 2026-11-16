"""Intelligent Diabetes Classification Using Archerfish Optimization and
Machine Learning — Flask application.

Run locally:
    python train_models.py     # one-time: AFO + model training
    python app.py              # http://127.0.0.1:5055

The application retrains automatically on first start if model artifacts are
missing, so it also works with a single command.
"""

from __future__ import annotations

import math
import os
import threading
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from database import db
from ml.data import FEATURE_COLUMNS, FEATURE_LABELS, FEATURE_UNITS, ZERO_AS_MISSING
from ml.service import PredictionService
from ml.trainer import ARTIFACT_PATH, MODEL_DISPLAY, train_and_evaluate

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = BASE_DIR / "database" / "diabetes.db"
APP_VERSION = "1.0.0"
TRAIN_LOCK = threading.Lock()

# Field definitions double as server-side validation and form metadata.
FIELD_SPECS = [
    {
        "name": "Pregnancies",
        "label": "Pregnancies",
        "unit": "count",
        "group": "profile",
        "min": 0,
        "max": 20,
        "step": 1,
        "placeholder": "2",
        "hint": "Number of times the patient has been pregnant.",
        "icon": "heart",
    },
    {
        "name": "Age",
        "label": "Age",
        "unit": "years",
        "group": "profile",
        "min": 1,
        "max": 120,
        "step": 1,
        "placeholder": "33",
        "hint": "Patient age in completed years.",
        "icon": "calendar",
    },
    {
        "name": "Glucose",
        "label": "Plasma Glucose",
        "unit": "mg/dL",
        "group": "lab",
        "min": 0,
        "max": 400,
        "step": 0.1,
        "placeholder": "117",
        "hint": "Plasma glucose concentration (2-hour oral glucose tolerance test in the source dataset).",
        "icon": "droplet",
    },
    {
        "name": "BloodPressure",
        "label": "Diastolic Blood Pressure",
        "unit": "mm Hg",
        "group": "lab",
        "min": 0,
        "max": 250,
        "step": 0.1,
        "placeholder": "72",
        "hint": "Diastolic blood pressure recorded during the assessment.",
        "icon": "activity",
    },
    {
        "name": "SkinThickness",
        "label": "Triceps Skin Thickness",
        "unit": "mm",
        "group": "lab",
        "min": 0,
        "max": 120,
        "step": 0.1,
        "placeholder": "23",
        "hint": "Triceps skinfold thickness. Zero is treated as missing.",
        "icon": "ruler",
    },
    {
        "name": "Insulin",
        "label": "Serum Insulin",
        "unit": "µU/mL",
        "group": "lab",
        "min": 0,
        "max": 1000,
        "step": 0.1,
        "placeholder": "30",
        "hint": "2-hour serum insulin. Zero is treated as missing.",
        "icon": "flask",
    },
    {
        "name": "BMI",
        "label": "Body Mass Index",
        "unit": "kg/m²",
        "group": "body",
        "min": 0,
        "max": 80,
        "step": 0.1,
        "placeholder": "32.0",
        "hint": "Body mass index (weight in kilograms divided by height in metres squared).",
        "icon": "scale",
    },
    {
        "name": "DiabetesPedigreeFunction",
        "label": "Diabetes Pedigree Function",
        "unit": "score",
        "group": "body",
        "min": 0,
        "max": 3,
        "step": 0.001,
        "placeholder": "0.372",
        "hint": "Family-history score expressing diabetes inheritance likelihood.",
        "icon": "git",
    },
]

MODEL_SHORT = {
    "svm": "SVM",
    "random_forest": "Random Forest",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
}

FORM_GROUPS = [
    {
        "key": "profile",
        "title": "Patient profile",
        "description": "Basic demographic and obstetric information.",
        "fields": [s for s in FIELD_SPECS if s["group"] == "profile"],
    },
    {
        "key": "lab",
        "title": "Laboratory measurements",
        "description": "Values from the patient's metabolic panel.",
        "fields": [s for s in FIELD_SPECS if s["group"] == "lab"],
    },
    {
        "key": "body",
        "title": "Body & family history",
        "description": "Anthropometric and hereditary risk indicators.",
        "fields": [s for s in FIELD_SPECS if s["group"] == "body"],
    },
]

FIELD_MAP = {spec["name"]: spec for spec in FIELD_SPECS}


def validate_patient_form(form) -> tuple:
    """Parse and validate the assessment form. Returns (values, errors)."""
    values = {}
    errors = {}
    for spec in FIELD_SPECS:
        raw = (form.get(spec["name"]) or "").strip()
        if raw == "":
            errors[spec["name"]] = "This field is required."
            continue
        try:
            value = float(raw)
        except ValueError:
            errors[spec["name"]] = "Enter a numeric value."
            continue
        if math.isnan(value) or math.isinf(value):
            errors[spec["name"]] = "Enter a valid number."
            continue
        if value < spec["min"] or value > spec["max"]:
            errors[spec["name"]] = "Enter a value between {} and {} {}.".format(
                spec["min"], spec["max"], spec["unit"]
            )
            continue
        values[spec["name"]] = value
    return values, errors


def ensure_model_service(app: Flask) -> None:
    """Load persisted artifacts; train once if they do not exist yet."""
    try:
        service = PredictionService()
    except FileNotFoundError:
        if not app.config.get("AUTO_TRAIN", True):
            raise
        app.logger.warning(
            "No model artifacts found — running the AFO training pipeline. "
            "This takes about 20 seconds on first start."
        )
        with TRAIN_LOCK:
            train_and_evaluate(verbose=False)
        service = PredictionService()
    app.extensions["prediction_service"] = service


def get_service() -> PredictionService:
    service = current_app.extensions.get("prediction_service")
    if service is None:
        abort(503, "Machine-learning service is unavailable.")
    return service


def create_app(config=None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "afo-diabetes-local-demo-key"),
        DATABASE=os.environ.get("DATABASE_PATH", str(DEFAULT_DATABASE)),
        AUTO_TRAIN=True,
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    )
    if config:
        app.config.update(config)

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    ensure_model_service(app)

    # ------------------------------------------------------------------ #
    # Template helpers
    # ------------------------------------------------------------------ #
    @app.template_filter("nice_datetime")
    def nice_datetime(value):
        try:
            parsed = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return value
        return parsed.strftime("%d %b %Y, %I:%M %p")

    @app.template_filter("pct")
    def pct(value, digits=1):
        try:
            return "{:.{}f}%".format(float(value) * 100.0, digits)
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("num")
    def num(value, digits=1):
        try:
            return "{:.{}f}".format(float(value), digits)
        except (TypeError, ValueError):
            return value

    @app.context_processor
    def inject_globals():
        service = current_app.extensions.get("prediction_service")
        metadata = service.metadata if service else {}
        return {
            "nav_date": datetime.now().strftime("%d %b %Y"),
            "model_meta": metadata,
            "site_name": "GlycoSense",
            "app_version": APP_VERSION,
            "model_display": MODEL_DISPLAY,
            "feature_labels": FEATURE_LABELS,
            "feature_units": FEATURE_UNITS,
        }

    # ------------------------------------------------------------------ #
    # Module 1 — Dashboard
    # ------------------------------------------------------------------ #
    @app.route("/")
    def dashboard():
        service = get_service()
        meta = service.metadata
        stats = db.get_stats()
        recent = db.recent_predictions(6)

        model_labels = [MODEL_SHORT[key] for key in MODEL_DISPLAY]
        baseline_scores = [
            meta["metrics"]["baseline"][key]["accuracy"] for key in MODEL_DISPLAY
        ]
        optimized_scores = [
            meta["metrics"]["optimized"][key]["accuracy"] for key in MODEL_DISPLAY
        ]

        return render_template(
            "dashboard.html",
            page="dashboard",
            stats=stats,
            recent=recent,
            meta=meta,
            chart_payload={
                "distribution": {
                    "labels": ["Diabetic", "Non-Diabetic"],
                    "values": [stats["diabetic"], stats["non_diabetic"]],
                },
                "models": {
                    "labels": model_labels,
                    "baseline": baseline_scores,
                    "optimized": optimized_scores,
                },
            },
        )

    # ------------------------------------------------------------------ #
    # Modules 2 & 5 — Assessment form and result
    # ------------------------------------------------------------------ #
    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        service = get_service()
        meta = service.metadata
        form_data = {}
        errors = {}

        if request.method == "POST":
            form_data = {
                **{spec["name"]: request.form.get(spec["name"], "") for spec in FIELD_SPECS},
                "patient_ref": request.form.get("patient_ref", ""),
            }
            values, errors = validate_patient_form(request.form)
            if not errors:
                try:
                    result = service.predict(values)
                    prediction_id = db.insert_prediction(
                        values,
                        result,
                        patient_ref=request.form.get("patient_ref", "")[:60],
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    current_app.logger.exception("Prediction failed")
                    flash(
                        "The prediction engine could not process this record: {}".format(exc),
                        "danger",
                    )
                else:
                    return redirect(url_for("result", prediction_id=prediction_id))

        return (
            render_template(
                "predict.html",
                page="predict",
                groups=FORM_GROUPS,
                form_data=form_data,
                errors=errors,
                meta=meta,
            ),
            400 if errors else 200,
        )

    @app.route("/result/<int:prediction_id>")
    def result(prediction_id):
        row = db.fetch_prediction(prediction_id)
        if row is None:
            abort(404)
        service = get_service()
        meta = service.metadata

        record = dict(row)
        inputs = []
        for spec in FIELD_SPECS:
            key = spec["name"]
            value = record[
                {
                    "Pregnancies": "pregnancies",
                    "Glucose": "glucose",
                    "BloodPressure": "blood_pressure",
                    "SkinThickness": "skin_thickness",
                    "Insulin": "insulin",
                    "BMI": "bmi",
                    "DiabetesPedigreeFunction": "dpf",
                    "Age": "age",
                }[key]
            ]
            inputs.append(
                {
                    "feature": key,
                    "label": spec["label"],
                    "unit": spec["unit"],
                    "value": value,
                    "display": format(value, "g"),
                    "imputed": key in ZERO_AS_MISSING and float(value) == 0.0,
                }
            )

        imputed = [item["label"] for item in inputs if item["imputed"]]
        result_data = {
            "prediction": record["prediction"],
            "label": "Diabetic" if record["prediction"] == 1 else "Non-Diabetic",
            "probability": record["probability"],
            "probability_percent": round(record["probability"] * 100.0, 1),
            "confidence_percent": round(
                max(record["probability"], 1.0 - record["probability"]) * 100.0, 1
            ),
            "risk_level": record["risk_level"],
            "risk_class": record["risk_level"].lower(),
            "model_name": record["model_name"],
            "selected_features": [
                feature.strip()
                for feature in (record["selected_features"] or "").split(",")
                if feature.strip()
            ],
            "threshold": meta.get("decision_threshold", 0.5),
        }

        return render_template(
            "result.html",
            page="predict",
            record=record,
            inputs=inputs,
            imputed=imputed,
            result=result_data,
            meta=meta,
        )

    # ------------------------------------------------------------------ #
    # Module 6 — Model performance
    # ------------------------------------------------------------------ #
    @app.route("/performance")
    def performance():
        meta = get_service().metadata
        model_keys = list(MODEL_DISPLAY)
        return render_template(
            "performance.html",
            page="performance",
            meta=meta,
            model_keys=model_keys,
            metrics=meta["metrics"],
            best=meta["metrics"]["optimized"][meta["best_model_key"]],
            best_baseline=meta["metrics"]["baseline"][meta["best_baseline_key"]],
            chart_payload={
                "models": {
                    "labels": [MODEL_SHORT[key] for key in model_keys],
                    "baseline": [
                        meta["metrics"]["baseline"][key]["accuracy"] for key in model_keys
                    ],
                    "optimized": [
                        meta["metrics"]["optimized"][key]["accuracy"] for key in model_keys
                    ],
                },
                "radar": {
                    "labels": ["Accuracy", "Precision", "Recall", "F1-score", "AUC"],
                    "series": [
                        {
                            "label": MODEL_DISPLAY[key],
                            "values": [
                                meta["metrics"]["optimized"][key]["accuracy"],
                                meta["metrics"]["optimized"][key]["precision"],
                                meta["metrics"]["optimized"][key]["recall"],
                                meta["metrics"]["optimized"][key]["f1"],
                                meta["metrics"]["optimized"][key]["roc_auc"],
                            ],
                        }
                        for key in model_keys
                    ],
                },
                "roc": [
                    {
                        "label": MODEL_DISPLAY[key],
                        "fpr": meta["metrics"]["optimized"][key]["roc"]["fpr"],
                        "tpr": meta["metrics"]["optimized"][key]["roc"]["tpr"],
                        "auc": meta["metrics"]["optimized"][key]["roc_auc"],
                    }
                    for key in model_keys
                ],
            },
        )

    # ------------------------------------------------------------------ #
    # Module 4 — AFO feature optimization
    # ------------------------------------------------------------------ #
    @app.route("/afo")
    def afo():
        meta = get_service().metadata
        return render_template(
            "afo.html",
            page="afo",
            meta=meta,
            afo=meta["afo"],
            metrics=meta["metrics"],
            model_keys=list(MODEL_DISPLAY),
            chart_payload={
                "convergence": meta["afo"]["convergence"],
                "baseline_fitness": meta["afo"]["baseline_fitness"],
                "frequency": meta["afo"]["feature_frequency"],
                "selected": meta["selected_features"],
                "labels": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
                "features": FEATURE_COLUMNS,
            },
        )

    @app.route("/afo/optimize", methods=["POST"])
    def rerun_afo():
        if not TRAIN_LOCK.acquire(blocking=False):
            flash("An optimization run is already in progress. Please wait.", "warning")
            return redirect(url_for("afo"))
        try:
            bundle = train_and_evaluate(verbose=False)
            current_app.extensions["prediction_service"] = PredictionService()
            flash(
                "AFO re-optimization complete: {} of {} features selected. "
                "Selected model: {}.".format(
                    len(bundle["selected_features"]),
                    len(FEATURE_COLUMNS),
                    bundle["best_model_name"],
                ),
                "success",
            )
        except Exception as exc:  # pragma: no cover - defensive
            current_app.logger.exception("AFO retraining failed")
            flash("Re-optimization failed: {}".format(exc), "danger")
        finally:
            TRAIN_LOCK.release()
        return redirect(url_for("afo"))

    # ------------------------------------------------------------------ #
    # Module 7 — Prediction history
    # ------------------------------------------------------------------ #
    @app.route("/history")
    def history():
        page = request.args.get("page", 1, type=int)
        per_page = 10
        query = (request.args.get("q") or "").strip()
        outcome = request.args.get("outcome") or ""
        model = request.args.get("model") or ""
        date_from = request.args.get("date_from") or ""
        date_to = request.args.get("date_to") or ""

        if outcome not in ("", "0", "1"):
            outcome = ""
        if model not in ("", *MODEL_DISPLAY.values()):
            model = ""

        rows, total, filters = db.list_predictions(
            page=page,
            per_page=per_page,
            query=query,
            outcome=outcome,
            model=model,
            date_from=date_from,
            date_to=date_to,
        )
        pages = max(1, math.ceil(total / per_page))
        if page > pages:
            page = pages
            rows, total, filters = db.list_predictions(
                page=page,
                per_page=per_page,
                query=query,
                outcome=outcome,
                model=model,
                date_from=date_from,
                date_to=date_to,
            )

        filtered_diabetic = sum(1 for row in rows if row["prediction"] == 1)
        return render_template(
            "history.html",
            page="history",
            rows=rows,
            total=total,
            filters=filters,
            pagination={
                "page": page,
                "pages": pages,
                "per_page": per_page,
                "has_prev": page > 1,
                "has_next": page < pages,
                "start": (page - 1) * per_page + 1 if total else 0,
                "end": min(page * per_page, total),
            },
            filtered_diabetic=filtered_diabetic,
            filtered_non_diabetic=len(rows) - filtered_diabetic,
        )

    # ------------------------------------------------------------------ #
    # Module 8 — System information
    # ------------------------------------------------------------------ #
    @app.route("/about")
    def about():
        meta = get_service().metadata
        return render_template("about.html", page="about", meta=meta)

    @app.route("/api/health")
    def health():
        meta = get_service().metadata
        return {
            "status": "ok",
            "model": meta["best_model_name"],
            "features": len(meta["selected_features"]),
            "accuracy": meta["best_metrics"]["accuracy"],
            "trained_at": meta["created_at"],
        }

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="Page not found"), 404

    @app.errorhandler(500)
    def server_error(_error):  # pragma: no cover - defensive
        return (
            render_template(
                "error.html",
                code=500,
                message="Something went wrong while processing this request.",
            ),
            500,
        )

    @app.cli.command("train")
    def train_command():
        """Retrain all classifiers and the AFO selector."""
        bundle = train_and_evaluate(verbose=True)
        print("Trained {} with {} features.".format(
            bundle["best_model_name"], len(bundle["selected_features"])
        ))

    return app


app = create_app()


if __name__ == "__main__":
    # Some environments export PORT=0 (meaning "pick any port"); this local
    # demonstration app always serves on a predictable address instead.
    port = int(os.environ.get("PORT") or "5055")
    if port == 0:
        port = 5055
    print("\n  GlycoSense — Intelligent Diabetes Classification")
    print("  Dashboard: http://127.0.0.1:{}\n".format(port))
    app.run(host="127.0.0.1", port=port, debug=False)
