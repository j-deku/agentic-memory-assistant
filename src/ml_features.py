import pandas as pd
from datetime import datetime


def build_features(df):

    # basic cleanup
    df = df.fillna(0)

    features = pd.DataFrame()

    # =========================
    # CATEGORY ENCODING
    # =========================
    features["category"] = df["category"].astype("category").cat.codes

    # =========================
    # TIME FEATURES
    # =========================
    features["created_hour"] = pd.to_datetime(df["created_at"]).dt.hour

    features["due_in_days"] = (
        pd.to_datetime(df["due_date"]) - pd.to_datetime(df["created_at"])
    ).dt.days

    # =========================
    # BEHAVIOR FEATURES
    # =========================
    features["completion_hour"] = df["completion_hour"].fillna(-1)

    features["completion_day"] = df["completion_day"].astype("category").cat.codes

    # =========================
    # TARGET LABEL
    # =========================
    target = df["completed"]

    return features, target