import sqlite3

DB = "conversation_memory.db"


def init_conversation_memory():

    conn = sqlite3.connect(DB)
    c = conn.cursor() 

    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_memory(key, value):

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO memories
        (key, value)
        VALUES (?, ?)
    """, (key, value))

    conn.commit()
    conn.close()


def get_memory(key):

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        "SELECT value FROM memories WHERE key=?",
        (key,)
    )

    row = c.fetchone()

    conn.close()

    return row[0] if row else None


def get_all_memories():

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("SELECT key, value FROM memories")

    rows = c.fetchall()

    conn.close()

    return rows