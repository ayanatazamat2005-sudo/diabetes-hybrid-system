from fastapi.testclient import TestClient
from src.decision import decide
from api.main import app

client = TestClient(app)


# ---- decision.py ----
def test_high_glucose_decided_by_rules_without_model(normal):
    r = decide({**normal, "Glucose": 215})
    assert r["decision"] == "high_risk"
    assert r["source"] == "pre_rules"
    assert r["probability"] is None


def test_invalid_input_never_reaches_model(normal):     # негативный
    r = decide({**normal, "Age": 250})
    assert r["decision"] == "invalid_input"
    assert r["probability"] is None


def test_model_probability_in_range(normal):
    r = decide(normal)
    assert 0.0 <= r["probability"] <= 1.0
    assert r["decision"] in {"low_risk", "high_risk", "expert_review"}


def test_risky_patient_is_not_low_risk():
    r = decide({"Pregnancies": 6, "Glucose": 170, "BloodPressure": 85, "SkinThickness": 35,
                "Insulin": 200, "BMI": 38, "DiabetesPedigreeFunction": 0.9, "Age": 50})
    assert r["decision"] in {"high_risk", "expert_review"}


# ---- API ----
def test_health_ok():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_returns_decision(normal):
    r = client.post("/predict", json=normal)
    assert r.status_code == 200
    assert {"decision", "label", "probability", "threshold"} <= set(r.json())


def test_explain_has_rules_and_features(normal):
    r = client.post("/explain", json={**normal, "Glucose": 150, "BMI": 32})
    body = r.json()
    assert r.status_code == 200
    assert "fired_rules" in body and len(body["top_features"]) == 3


def test_predict_impossible_age_returns_422(normal):    # негативный
    r = client.post("/predict", json={**normal, "Age": 250})
    assert r.status_code == 422 and "errors" in r.json()["detail"]


def test_predict_wrong_type_returns_422(normal):        # негативный
    assert client.post("/predict", json={**normal, "Age": "abc"}).status_code == 422


def test_predict_missing_glucose_returns_422(normal):   # негативный
    d = dict(normal)
    d.pop("Glucose")
    assert client.post("/predict", json=d).status_code == 422
