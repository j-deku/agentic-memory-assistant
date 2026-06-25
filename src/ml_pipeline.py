import pandas as pd

# All feature engineering now lives in ml_dataset.load_training_data().
# By the time build_features() is called, the DataFrame already has the
# four final columns: category_encoded, created_hour, due_in_days, is_completed.
# This function is kept so classifier.py doesn't need to change its import.

FEATURE_COLUMNS = [
    "category_encoded",
    "created_hour",
    "due_in_days",
    "is_completed",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_features: DataFrame is missing columns: {missing}. "
            "Make sure you're loading data via ml_dataset.load_training_data()."
        )
    return df[FEATURE_COLUMNS].copy()