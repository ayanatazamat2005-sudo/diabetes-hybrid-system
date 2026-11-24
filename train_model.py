"""Переобучение модели под версию scikit-learn на этом компьютере (то же, что в ноутбуке)."""
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from src.preprocessing import zeros_to_nan

df = pd.read_csv("data/diabetes.csv")
FEATURES = [c for c in df.columns if c != "Outcome"]
X, y = df[FEATURES], df["Outcome"]
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("zeros", FunctionTransformer(zeros_to_nan)), ("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler()),
                 ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000))])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
proba = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
ths = np.arange(0.1, 0.9, 0.01)
thr = float(max(t for t in ths if recall_score(y_tr, proba >= t) >= 0.85))   # Recall >= 0.85

pipe.fit(X, y)
joblib.dump({"pipeline": pipe, "threshold": thr, "features": FEATURES}, "models/model.joblib")
print("Модель сохранена, порог =", round(thr, 3))
