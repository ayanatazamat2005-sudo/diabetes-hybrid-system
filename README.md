# Гибридная система: риск диабета (вариант 3, Pima Indians Diabetes)

Экспертные правила (R1–R18) + логистическая регрессия + FastAPI + Streamlit.

## Запуск (3 команды)
```
pip install -r requirements.txt
uvicorn api.main:app --reload          # API: http://127.0.0.1:8000/docs
streamlit run app/ui.py                # интерфейс (в другом терминале)
```
Тесты: `python -m pytest -q`. Модель обучается в `notebooks/eda_model.ipynb`, датасет: `data/diabetes.csv` (Kaggle).
