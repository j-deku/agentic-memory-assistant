"""
ml_dataset.py  (fixed)

Produces a DataFrame with the exact columns the trainer expects:
  category_encoded, created_hour, due_in_days, is_completed

The old build_ml_dataset() in tasks.py used different column names
(task_length, category, days_left, is_overdue, label_completed) which
caused the KeyError crash in train_all_models().
"""

import pandas as pd
from datetime import datetime
from database import get_connection


# Category → integer mapping.  Add new categories at the end to avoid
# shifting existing encoded values and invalidating saved model weights.
CATEGORY_MAP = {
    "work":       0,
    "personal":   1,
    "health":     2,
    "shopping":   3,
    "finance":    4,
    "study":      5,
    "school":     5,   # alias for study
    "learning":   5,   # alias for study
    "other":      6,
}

def _encode_category(cat: str) -> int:
    return CATEGORY_MAP.get((cat or "other").lower().strip(), 6)


def load_training_data() -> pd.DataFrame:
    """
    Reads the tasks table and returns a clean DataFrame with columns:
        category_encoded  int   — encoded category
        created_hour      int   — hour of day the task was created (0-23)
        due_in_days       int   — days between creation and due date (can be negative)
        is_completed      int   — 1 if done, 0 if not

    Rows with missing due_date or created_at are skipped gracefully.
    Prints the dataset size so you can see it in the console.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, category, due_date, completed, created_at FROM tasks")
    rows = cursor.fetchall()
    conn.close()

    records = []
    today = datetime.now().date()

    for row in rows:
        task_id, title, category, due_date_str, completed, created_at_str = row

        # ── due_in_days ──────────────────────────────────────
        try:
            due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            # No due date — use today as a neutral value
            due_date = today
        due_in_days = (due_date - today).days

        # ── created_hour ─────────────────────────────────────
        try:
            created_hour = datetime.fromisoformat(str(created_at_str)).hour
        except (TypeError, ValueError):
            created_hour = 12  # neutral default

        records.append({
            "category_encoded": _encode_category(category),
            "created_hour":     created_hour,
            "due_in_days":      due_in_days,
            "is_completed":     int(completed or 0),
        })

    df = pd.DataFrame(records, columns=[
        "category_encoded", "created_hour", "due_in_days", "is_completed"
    ])

    print(f"Dataset size: {len(df)}")
    return df