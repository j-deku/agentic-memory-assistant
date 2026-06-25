import joblib
from database import get_connection
import pandas as pd
import os

MODEL_PATH = "task_model.pkl"


def get_dataset_size():
    from ml_dataset import load_training_data
    df = load_training_data()
    return len(df)


def get_average_error():
    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT error
        FROM predictions
        WHERE error IS NOT NULL
    """, conn)

    conn.close()

    if df.empty:
        return 0

    return df["error"].mean()


def model_needs_retrain():
    """
    Decision logic for retraining
    """

    size = get_dataset_size()
    error = get_average_error()

    # conditions for retrain
    if size < 10:
        return False, "Not enough data"

    if error > 0.35:
        return True, "High prediction error"

    if size % 5 == 0:
        return True, "New data batch reached"

    return False, "Model stable"