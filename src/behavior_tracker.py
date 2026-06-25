from database import get_connection
import pandas as pd


def load_tasks():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM tasks", conn)
    conn.close()
    return df


# =========================
# USER BEHAVIOR PROFILE
# =========================

def build_behavior_profile():
    df = load_tasks()

    if df.empty:
        return None

    completed = df[df["completed"] == 1]

    profile = {}

    # best working hour
    if not completed.empty:
        profile["best_hour"] = int(
            completed["completion_hour"].mode()[0]
        )

        profile["best_day"] = completed["completion_day"].mode()[0]

    # completion rate
    profile["completion_rate"] = (
        df["completed"].sum() / len(df)
    ) * 100

    # category behavior
    category_success = df.groupby("category")["completed"].mean()

    profile["category_performance"] = category_success.to_dict()

    return profile