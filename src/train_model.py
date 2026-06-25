from sklearn.ensemble import RandomForestClassifier
import joblib

from ml_dataset import load_training_data
from ml_pipeline import build_features


def train_model():

    df = load_training_data()

    if len(df) < 5:
        print("Not enough data to train yet.")
        return

    # TARGET
    y = df["completed"]

    # FEATURES (NOW CONSISTENT)
    X = build_features(df)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "task_model.pkl")

    print("Model trained successfully ✔")

    print("Dataset size:", len(df))

if __name__ == "__main__":
        train_model()