import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.decision import decide, load_bundle
from src.explain import top_features

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "logs" / "decisions.db"

app = FastAPI(title="Diabetes Hybrid System", version="1.0")


class PatientInput(BaseModel):
    Pregnancies: Optional[float] = None
    Glucose: Optional[float] = None
    BloodPressure: Optional[float] = None
    SkinThickness: Optional[float] = None
    Insulin: Optional[float] = None
    BMI: Optional[float] = None
    DiabetesPedigreeFunction: Optional[float] = None
    Age: Optional[float] = None


# ---------- журнал ----------
def init_db():
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, input TEXT, decision TEXT, source TEXT, probability REAL)""")

def log_decision(data, result):
    with sqlite3.connect(DB) as con:
        con.execute(
            "INSERT INTO decisions (ts, input, decision, source, probability) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"),
             json.dumps(data), result["decision"], result["source"], result["probability"]))

init_db()


# ---------- общий запуск ----------
def _run(patient: PatientInput):
    data = patient.model_dump()
    result = decide(data)
    log_decision(data, result)
    if result["decision"] == "invalid_input":
        raise HTTPException(status_code=422, detail={
            "message": result["label"],
            "errors": [{"rule": e["rule"], "message": e["message"]} for e in result["errors"]]})
    return data, result


# ---------- эндпоинты ----------
@app.get("/health")
def health():
    try:
        b = load_bundle()
        return {"status": "ok", "model_loaded": True, "threshold": round(b["threshold"], 3)}
    except Exception as e:
        return {"status": "error", "model_loaded": False, "detail": str(e)}


@app.post("/predict")
def predict(patient: PatientInput):
    data, r = _run(patient)
    return {"decision": r["decision"], "label": r["label"], "source": r["source"],
            "probability": r["probability"], "threshold": r["threshold"],
            "warnings": [w["message"] for w in r["warnings"]]}


@app.post("/explain")
def explain(patient: PatientInput):
    data, r = _run(patient)
    return {"decision": r["decision"], "label": r["label"], "source": r["source"],
            "probability": r["probability"],
            "fired_rules": [{"rule": x["rule"], "group": x["group"],
                             "message": x["message"], "source": x["source"]}
                            for x in r["rules"]],
            "top_features": top_features(data) if r["probability"] is not None else [],
            "warnings": [w["message"] for w in r["warnings"]]}
