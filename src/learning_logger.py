from database import get_connection

def log_event(task_id, event_type, reward):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO learning_events (
        task_id,
        event_type,
        reward
    )
    VALUES (?, ?, ?)
    """, (
        task_id,
        event_type,
        reward
    ))

    conn.commit()
    conn.close()