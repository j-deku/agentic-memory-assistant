from sklearn.ensemble import RandomForestRegressor
import joblib


def train_regressor(df):
    # "computed_time_to_complete" is only present once you start tracking
    # actual_minutes in the tasks table. Until then, skip silently.
    if "computed_time_to_complete" not in df.columns:
        return

    df = df.dropna(subset=["computed_time_to_complete"])

    if len(df) < 3:
        return

    y = df["computed_time_to_complete"]
    X = df[[
        "category_encoded",
        "created_hour",
        "due_in_days",
        "is_completed"
    ]]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "duration_model.pkl")
    print("Regressor trained ✔")