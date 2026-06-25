import sqlite3
import pandas as pd

DB = "rl_memory.db"


def load_rl_data():
    conn = sqlite3.connect(DB)

    df = pd.read_sql_query("""
        SELECT * FROM rl_events
    """, conn)

    conn.close()

    return df


def compute_weights(df):
    df["weight"] = df["reward"].apply(lambda x: 2.0 if x > 0 else 0.5)
    return df