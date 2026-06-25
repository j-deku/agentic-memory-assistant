from database import get_connection
import json


def create_agent_events_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def log_agent_event(event_type, data):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO agent_events (
        event_type,
        data
    )
    VALUES (?, ?)
    """, (
        event_type,
        json.dumps(data)
    ))

    conn.commit()
    conn.close()