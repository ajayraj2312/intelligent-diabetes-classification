"""Focused integration tests for the GlycoSense Flask application.

Run with:
    .venv/bin/pytest -q
"""

from __future__ import annotations

import pytest

from app import create_app
from database import db
from ml.data import FEATURE_COLUMNS

VALID_FORM = {
    "patient_ref": "TEST-PATIENT",
    "Pregnancies": "6",
    "Glucose": "148",
    "BloodPressure": "72",
    "SkinThickness": "35",
    "Insulin": "0",
    "BMI": "33.6",
    "DiabetesPedigreeFunction": "0.627",
    "Age": "50",
}


@pytest.fixture()
def app(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "test.db"),
            "AUTO_TRAIN": False,
            "SECRET_KEY": "test-secret",
        }
    )
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.mark.parametrize(
    "path",
    ["/", "/predict", "/performance", "/afo", "/history", "/about"],
)
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"GlycoSense" in response.data


def test_unknown_page_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_invalid_form_is_rejected(client):
    response = client.post(
        "/predict",
        data={**VALID_FORM, "Glucose": "", "Age": "999"},
    )
    assert response.status_code == 400
    assert b"This field is required." in response.data
    assert b"Enter a value between" in response.data


def test_prediction_is_saved_and_displayed(app, client):
    response = client.post("/predict", data=VALID_FORM)
    assert response.status_code == 302

    with app.app_context():
        row = db.get_db().execute("SELECT * FROM predictions").fetchone()
        assert row is not None
        assert row["patient_ref"] == "TEST-PATIENT"
        assert row["prediction"] in (0, 1)
        assert 0.0 <= row["probability"] <= 1.0
        prediction_id = row["id"]

    result = client.get("/result/{}".format(prediction_id))
    assert result.status_code == 200
    assert b"Medical disclaimer" in result.data
    assert b"diabetes probability" in result.data


def test_history_search_and_filter(app, client):
    client.post("/predict", data=VALID_FORM)

    with app.app_context():
        row = db.get_db().execute("SELECT prediction FROM predictions").fetchone()
        outcome = str(row["prediction"])

    matching = client.get("/history?q=TEST-PATIENT&outcome={}".format(outcome))
    assert matching.status_code == 200
    assert b"TEST-PATIENT" in matching.data

    non_matching = client.get("/history?q=NOT-A-REAL-REFERENCE")
    assert non_matching.status_code == 200
    assert b"No matching records" in non_matching.data


def test_model_service_uses_selected_features(app):
    with app.app_context():
        service = app.extensions["prediction_service"]
        assert service.is_ready
        assert 2 <= len(service.bundle["selected_features"]) < len(FEATURE_COLUMNS)
        assert "roc_auc" in service.bundle["best_metrics"]

        values = {feature: float(VALID_FORM[feature]) for feature in FEATURE_COLUMNS}
        result = service.predict(values)
        assert result["label"] in ("Diabetic", "Non-Diabetic")
        assert 0.0 <= result["probability"] <= 1.0
        assert result["model_name"]


def test_health_payload(client):
    payload = client.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert 0.0 < payload["accuracy"] <= 1.0


def test_afo_retrain_route(app, client, monkeypatch):
    """Exercise the Re-optimize button without repeating the 20s training run."""
    import joblib
    import app as app_module
    from ml.trainer import ARTIFACT_PATH

    bundle = joblib.load(ARTIFACT_PATH)
    monkeypatch.setattr(
        app_module, "train_and_evaluate", lambda verbose=False: bundle
    )
    response = client.post("/afo/optimize", follow_redirects=True)
    assert response.status_code == 200
    assert b"AFO re-optimization complete" in response.data
