"""База знаний (Pima Indians Diabetes).
Группа А: жёсткие ограничения (pre_rules) - решение без модели
Группа Б: проверка данных (validate_input)
Группа В: постправила (post_rules) - рекомендации и коррекция модели
"""

REQUIRED = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
CRITICAL = ["Glucose", "Age"]          # без них решение невозможно

# Поля, где 0 физиологически невозможен -> скрытый пропуск
ZERO_MEANS_MISSING = {"Glucose": "R5", "BloodPressure": "R6", "BMI": "R7",
                      "SkinThickness": "R8", "Insulin": "R9"}

# Физиологически возможные диапазоны
LIMITS = {"Pregnancies": (0, 20), "Glucose": (20, 600),
          "BloodPressure": (30, 250), "SkinThickness": (1, 100),
          "Insulin": (1, 1000), "BMI": (10, 80),
          "DiabetesPedigreeFunction": (0.0, 3.0), "Age": (1, 120)}


def _rule(rid, group, message, source, decision=None):
    return {"rule": rid, "group": group, "message": message,
            "source": source, "decision": decision}


# ---------- Группа Б: проверка данных (R4-R11) ----------
def validate_input(data):
    clean, errors, warnings = {}, [], []
    for f in REQUIRED:
        v = data.get(f)
        if v is None or v == "":                       # R4
            clean[f] = None
            if f in CRITICAL:
                errors.append(_rule("R4", "validation",
                    f"Обязательное поле {f} не заполнено.", "Проектное решение"))
            else:
                warnings.append(_rule("R4", "validation",
                    f"Поле {f} пустое: значение будет заполнено медианой в Pipeline.",
                    "Проектное решение"))
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):   # R11
            clean[f] = None
            errors.append(_rule("R11", "validation",
                f"Поле {f} должно быть числом, получено: {v!r}.", "Проектное решение"))
            continue
        if f in ZERO_MEANS_MISSING and v == 0:        # R5-R9
            clean[f] = None
            warnings.append(_rule(ZERO_MEANS_MISSING[f], "validation",
                f"{f} = 0 физиологически невозможно: считаем пропуском.",
                "Особенность датасета Pima"))
            continue
        lo, hi = LIMITS[f]
        if not (lo <= v <= hi):                        # R10
            clean[f] = None
            errors.append(_rule("R10", "validation",
                f"{f} = {v} вне допустимого диапазона [{lo}, {hi}].",
                "Физиологические пределы"))
            continue
        clean[f] = v
    return clean, errors, warnings


# ---------- Группа А: жёсткие ограничения (R1-R3) ----------
def pre_rules(d):
    out = []
    g, bp, age = d.get("Glucose"), d.get("BloodPressure"), d.get("Age")
    if g is not None and g >= 200:                     # R1
        out.append(_rule("R1", "hard",
            "Глюкоза через 2 ч в ОГТТ >= 200 мг/дл: критерий диабета, требуется врач.",
            "ВОЗ, критерии диагностики диабета", "high_risk"))
    if bp is not None and bp >= 120:                   # R2
        out.append(_rule("R2", "hard",
            "Диастолическое давление >= 120 мм рт. ст.: срочная консультация врача.",
            "AHA/ACC, гипертонический криз", "expert_review"))
    if age is not None and age < 18:                   # R3
        out.append(_rule("R3", "hard",
            "Модель обучена на взрослых (21+): для возраста < 18 нужен специалист.",
            "Ограничение датасета", "expert_review"))
    return out


# ---------- Группа В: постправила (R12-R18) ----------
def post_rules(d, probability=None, threshold=0.5):
    out = []
    g, bmi, age = d.get("Glucose"), d.get("BMI"), d.get("Age")
    bp, dpf = d.get("BloodPressure"), d.get("DiabetesPedigreeFunction")
    low_prob = probability is not None and probability < threshold

    if g is not None and 140 <= g < 200:               # R12
        out.append(_rule("R12", "post",
            "Нарушенная толерантность к глюкозе (140-199): рекомендуется подтверждающий анализ и консультация.",
            "ВОЗ, IGT", "expert_review" if low_prob else None))
    if bmi is not None and bmi >= 30:                  # R13
        out.append(_rule("R13", "post",
            "ИМТ >= 30 (ожирение): снижение веса и физическая активность (150 мин/нед).",
            "ВОЗ, классификация ИМТ"))
    elif bmi is not None and 25 <= bmi < 30:           # R14
        out.append(_rule("R14", "post",
            "ИМТ 25-29.9 (избыточный вес): рекомендуется коррекция питания и активность.",
            "ВОЗ, классификация ИМТ"))
    if age is not None and age >= 45:                  # R15
        out.append(_rule("R15", "post",
            "Возраст >= 45: рекомендуется регулярный скрининг на диабет.",
            "ADA, Standards of Care"))
    if bp is not None and bp >= 90:                    # R16
        out.append(_rule("R16", "post",
            "Диастолическое давление >= 90: повышенное, требуется оценка у врача.",
            "ESC/ESH, гипертензия"))
    if dpf is not None and dpf >= 1.0:                 # R17
        out.append(_rule("R17", "post",
            "Высокая наследственная предрасположенность (DPF >= 1.0): регулярные проверки.",
            "Решение эксперта проекта"))
    if (low_prob and bmi is not None and bmi >= 30
            and age is not None and age >= 45):        # R18
        out.append(_rule("R18", "post",
            "Модель даёт низкий риск, но ИМТ >= 30 и возраст >= 45: нужна проверка специалиста.",
            "Коррекция модели (конфликт ML и правил)", "expert_review"))
    return out


# ---------- Общий запуск ----------
def check_rules(data, probability=None, threshold=0.5):
    clean, errors, warnings = validate_input(data)
    pre = pre_rules(clean) if not errors else []
    post = post_rules(clean, probability, threshold) if not errors else []
    return {"clean": clean, "errors": errors, "warnings": warnings,
            "pre": pre, "post": post}
