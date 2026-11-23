import numpy as np

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def zeros_to_nan(X):
    X = X.copy()
    for col in ZERO_AS_MISSING:
        X[col] = X[col].replace(0, np.nan)
    return X
