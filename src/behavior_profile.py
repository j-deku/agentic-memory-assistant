from database import get_connection
import pandas as pd


def build_behavior_profile():

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT *
        FROM tasks
    """, conn)

    conn.close()

    if df.empty:
        return {}

    profile = {}

    profile["completion_rate"] = float(
        df["completed"].mean()
    )

    profile["category_performance"] = {
    k: float(v)
    for k, v in (
        df.groupby("category")["completed"]
        .mean()
        .to_dict()
        .items()
    )
}

    completed = df[df["completion_hour"].notna()]

    if not completed.empty:
        profile["best_hour"] = int(
            completed["completion_hour"].mode()[0]
        )

    return profile