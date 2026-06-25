from sklearn.ensemble import RandomForestClassifier
import joblib
from ml_pipeline import build_features


def train_classifier(df):
    # Column produced by ml_dataset.py is "is_completed", not "completed"
    y = df["is_completed"]
    X = build_features(df)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "completion_model.pkl")
    print("Classifier trained ✔")