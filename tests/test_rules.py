from src.rules import validate_input, pre_rules, post_rules


def _pre_ids(data):
    clean, _, _ = validate_input(data)
    return [r["rule"] for r in pre_rules(clean)]


def test_glucose_200_triggers_R1(normal):
    assert "R1" in _pre_ids({**normal, "Glucose": 200})


def test_glucose_199_does_not_trigger_R1(normal):      # граничное значение
    assert "R1" not in _pre_ids({**normal, "Glucose": 199})


def test_zero_glucose_is_hidden_missing(normal):
    clean, errors, warnings = validate_input({**normal, "Glucose": 0})
    assert clean["Glucose"] is None
    assert "R5" in [w["rule"] for w in warnings]


def test_obesity_rule_R13(normal):
    assert "R13" in [r["rule"] for r in post_rules({**normal, "BMI": 32})]


def test_model_conflict_rule_R18(normal):
    d = {**normal, "BMI": 31, "Age": 50}
    out = post_rules(d, probability=0.2, threshold=0.376)
    r18 = [r for r in out if r["rule"] == "R18"]
    assert r18 and r18[0]["decision"] == "expert_review"


# ---- негативные тесты ----
def test_impossible_age_rejected(normal):
    _, errors, _ = validate_input({**normal, "Age": 250})
    assert [e["rule"] for e in errors] == ["R10"]


def test_text_in_numeric_field_rejected(normal):
    _, errors, _ = validate_input({**normal, "BMI": "abc"})
    assert errors[0]["rule"] == "R11"


def test_missing_required_age_rejected(normal):
    d = dict(normal)
    d.pop("Age")
    _, errors, _ = validate_input(d)
    assert errors[0]["rule"] == "R4"
