import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from feature_schema import task_to_features


MODEL_PATH = "task_model.pkl"


def train_model():
    df = pd.read_csv("ml_tasks_dataset.csv")

    if df.empty:
        print("No data to train.")
        return

    # =========================
    # FEATURE ENGINEERING (SINGLE SOURCE OF TRUTH)
    # =========================
    X = df.drop("label_completed", axis=1)
    y = df["label_completed"]

    categorical_features = ["category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"Model trained ✔ Accuracy: {accuracy:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved ✔ {MODEL_PATH}")

if __name__ == "__main__":
    train_model()

def predict_task(task):
    """
    Always uses SAME feature schema as training via task_to_features()
    """

    model = joblib.load(MODEL_PATH)

    features = task_to_features(task)

    input_data = pd.DataFrame([features])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability
