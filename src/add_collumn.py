import sqlite3

conn = sqlite3.connect("database/tasks.db")
c = conn.cursor()

c.execute("""
ALTER TABLE tasks ADD COLUMN time_to_complete REAL
""")

conn.commit()
conn.close()

print("DB fixed")