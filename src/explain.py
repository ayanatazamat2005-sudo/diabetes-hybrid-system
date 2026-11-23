import pandas as pd
from src.decision import load_bundle
from src.rules import validate_input


def top_features(data, n=3):
    bundle = load_bundle()
    clean, _, _ = validate_input(data)
    X = pd.DataFrame([clean], columns=bundle["features"]).astype(float)
    pipe = bundle["pipeline"]
    X_scaled = pipe[:-1].transform(X)          # zeros -> imputer -> scaler
    coefs = pipe.named_steps["model"].coef_[0]
    contrib = X_scaled[0] * coefs
    items = sorted(zip(bundle["features"], contrib),
                   key=lambda t: abs(t[1]), reverse=True)[:n]
    return [{"feature": f,
             "contribution": round(float(c), 3),
             "effect": "повышает риск" if c > 0 else "снижает риск"}
            for f, c in items]
