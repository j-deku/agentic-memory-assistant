from database import get_connection


# =========================
# 1. GET WEIGHT
# =========================

def get_weight(key, default=1.0):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM learning_weights WHERE key = ?", (key,))
    row = cursor.fetchone()

    conn.close()

    return row[0] if row else default


# =========================
# 2. UPDATE WEIGHT
# =========================

def update_weight(key, value):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO learning_weights (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = ?
    """, (key, value, value))

    conn.commit()
    conn.close()


# =========================
# 3. LEARN FROM OUTCOME
# =========================

def learn_from_task(task, predicted_score, actually_completed):
    category = task[2].lower()

    weight_key = f"category_{category}_weight"

    current_weight = get_weight(weight_key, 1.0)

    # =========================
    # SUCCESS CASE
    # =========================
    if actually_completed and predicted_score >= 50:
        new_weight = current_weight + 0.1

    # =========================
    # FAILURE CASE
    # =========================
    elif not actually_completed and predicted_score >= 50:
        new_weight = current_weight - 0.1

    else:
        new_weight = current_weight

    # clamp values
    new_weight = max(0.1, min(2.0, new_weight))

    update_weight(weight_key, new_weight)

    return new_weight