"""Гибридный модуль решения: правила -> ML -> постправила."""
import joblib
import pandas as pd
from pathlib import Path
from src.rules import validate_input, pre_rules, post_rules

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
UNCERTAIN_MARGIN = 0.07      # зона неуверенности вокруг порога: [thr-0.07; thr+0.07]

LABELS = {
    "high_risk": "Высокий риск диабета: рекомендуется обратиться к врачу.",
    "low_risk": "Низкий риск диабета по заданным данным.",
    "expert_review": "Требуется проверка специалиста.",
    "invalid_input": "Данные некорректны: решение не принято.",
}

_bundle = None


def load_bundle(path=MODEL_PATH):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def _finish(result, decision, source):
    result["decision"] = decision
    result["label"] = LABELS[decision]
    result["source"] = source
    return result


def decide(data):
    clean, errors, warnings = validate_input(data)
    result = {"decision": None, "label": None, "source": None,
              "probability": None, "threshold": None,
              "rules": [], "warnings": warnings, "errors": errors}

    # 1) некорректные данные -> модель не вызываем
    if errors:
        return _finish(result, "invalid_input", "validation")

    # 2) жёсткие правила (группа А) -> решение без модели
    pre = pre_rules(clean)
    if pre:
        result["rules"] = pre
        high = any(r["decision"] == "high_risk" for r in pre)
        return _finish(result, "high_risk" if high else "expert_review", "pre_rules")

    # 3) модель
    bundle = load_bundle()
    X = pd.DataFrame([clean], columns=bundle["features"]).astype(float)
    proba = float(bundle["pipeline"].predict_proba(X)[0, 1])
    thr = bundle["threshold"]
    result["probability"] = round(proba, 3)
    result["threshold"] = round(thr, 3)

    if abs(proba - thr) <= UNCERTAIN_MARGIN:
        decision, source = "expert_review", "model_uncertain"
    elif proba >= thr:
        decision, source = "high_risk", "model"
    else:
        decision, source = "low_risk", "model"

    # 4) постправила (группа В): рекомендации и коррекция
    post = post_rules(clean, proba, thr)
    result["rules"] = post
    if decision == "low_risk" and any(r["decision"] == "expert_review" for r in post):
        decision, source = "expert_review", "post_rules"

    return _finish(result, decision, source)
