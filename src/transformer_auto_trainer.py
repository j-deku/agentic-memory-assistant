# transformer_auto_trainer.py

import sqlite3
import time
from transformer_train import train_transformer

DB = "rl_memory.db"
_last_train_time = 0

def get_rl_event_count():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM rl_events")
    count = c.fetchone()[0]

    conn.close()

    return count


def retrain_transformer_if_needed(step=4, cooldown=300):
    global _last_train_time

    now = time.time()
    count = get_rl_event_count()

    if count == 0:
        return False

    if count % step != 0:
        return False

    if now - _last_train_time < cooldown:
        return False
    
    print(f"\n🧠 Retraining Transformer ({count} experiences)...")

    train_transformer()

    _last_train_time = now
    return True