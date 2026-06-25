import sqlite3
from datetime import datetime

DB = "rl_memory.db"


def init_rl_memory():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS rl_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            predicted_intent TEXT,
            actual_intent TEXT,
            reward REAL,
            confidence REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

def update_rl_memory(task_id, reward, confidence=1.0):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        INSERT INTO rl_events (text, predicted_intent, actual_intent, reward, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(task_id),
        "task_system",
        "task_completed",
        reward,
        confidence,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()