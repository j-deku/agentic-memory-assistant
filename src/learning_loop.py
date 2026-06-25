from tasks import list_tasks
from ml_dataset import load_training_data
from train_model import train_model


def retrain_if_needed():
    df = load_training_data()

    completed = df["completed"].sum()
    total = len(df)

    if total < 10:
        return "Not enough data yet"

    # retrain condition (simple version)
    if completed / total > 0.3:
        train_model()
        return "Model retrained with new behavior data"

    return "No retraining needed"