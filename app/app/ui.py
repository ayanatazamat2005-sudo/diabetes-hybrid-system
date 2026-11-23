import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Оценка риска диабета", page_icon="🩺")
st.title("🩺 Оценка риска диабета")
st.caption("Гибридная система: экспертные правила + машинное обучение. Не заменяет врача.")

FIELDS = [
    ("Pregnancies", "Количество беременностей"),
    ("Glucose", "Глюкоза через 2 ч после нагрузки (мг/дл)"),
    ("BloodPressure", "Диастолическое давление (мм рт. ст.)"),
    ("SkinThickness", "Толщина кожной складки (мм)"),
    ("Insulin", "Инсулин через 2 ч (мкЕд/мл)"),
    ("BMI", "ИМТ"),
    ("DiabetesPedigreeFunction", "Наследственный индекс диабета"),
    ("Age", "Возраст (лет)"),
]

with st.form("patient"):
    values = {}
    for name, label in FIELDS:
        values[name] = st.number_input(label, value=None, placeholder="не указано")
    submitted = st.form_submit_button("Оценить риск")

if submitted:
    try:
        resp = requests.post(f"{API_URL}/explain", json=values, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error("API не запущен. Запустите: python -m uvicorn api.main:app --reload")
        st.stop()

    if resp.status_code == 422:
        detail = resp.json().get("detail")
        st.error("Данные некорректны, решение не принято.")
        if isinstance(detail, dict):
            for e in detail["errors"]:
                st.write(f"• [{e['rule']}] {e['message']}")
        else:
            st.write("Проверьте тип значений: нужны числа.")
        st.stop()

    r = resp.json()
    box = {"high_risk": st.error, "low_risk": st.success, "expert_review": st.warning}
    box.get(r["decision"], st.info)(r["label"])

    if r["probability"] is not None:
        thr = r.get("threshold")
        help_text = f"Порог решения: {thr:.0%}" if thr is not None else None
        st.metric("Вероятность диабета", f"{r['probability']:.0%}", help=help_text)
        st.progress(min(max(r["probability"], 0.0), 1.0))
    else:
        st.caption("Решение принято по жёстким правилам, без модели.")

    for w in r["warnings"]:
        st.info(w)

    st.subheader("Почему такое решение")
    if r["fired_rules"]:
        for x in r["fired_rules"]:
            st.write(f"**{x['rule']}**: {x['message']}  \n_{x['source']}_")
    else:
        st.write("Дополнительные правила не сработали.")

    if r["top_features"]:
        st.subheader("Признаки, сильнее всего повлиявшие на модель")
        for f in r["top_features"]:
            st.write(f"• {f['feature']}: {f['effect']} ({f['contribution']:+.2f})")