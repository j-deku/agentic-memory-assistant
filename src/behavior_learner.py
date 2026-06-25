from database import get_connection
import pandas as pd

def learn_behavior():

    conn = get_connection()

    df = pd.read_sql_query("""
        SELECT *
        FROM learning_events
    """, conn)

    conn.close()

    if df.empty:
        return {}

    avg_reward = df["reward"].mean()

    return {
        "avg_reward": avg_reward,
        "events": len(df)
    }