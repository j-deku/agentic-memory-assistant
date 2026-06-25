import joblib
import pandas as pd
from pathlib import Path
from ml_pipeline import build_features

MODEL_PATH = Path("task_model.pkl")

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict_task_completion(task):

    if model is None:
        return 0.5  # neutral guess

    df = pd.DataFrame([task])
    X = build_features(df)

    try:
        prob = model.predict_proba(X)[0][1]
        return float(prob)
    except:
        return 0.5