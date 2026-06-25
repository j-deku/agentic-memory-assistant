from database import get_connection
import pandas as pd

def load_data():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM tasks", conn)
    conn.close()
    return df


def productivity_summary():
    df = load_data()

    if df.empty:
        return "No data yet."

    total = len(df)
    completed = df["completed"].sum()
    rate = (completed / total) * 100

    return f"""
Total tasks: {total}
Completed: {completed}
Completion rate: {rate:.2f}%
"""