from ml_model import predict_task
from transformer_predict import predict_task_transformer
from tasks import get_task_score, get_memory
from datetime import datetime
from transformer_predict import predict_transformer_score

def normalize_score(value, max_value=100):
    return value / max_value

from datetime import datetime

def safe_parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        # fallback for bad formats like "2027-02"
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except:
            return datetime.now().date()


def safe_transformer_predict(task):
    """
    Safe wrapper so system never crashes if transformer fails
    """
    try:
        return predict_task_transformer(task)
    except Exception:
        return 0.5  # neutral score


def hybrid_decision(task):

    # =========================
    # CORE TASK DATA
    # =========================
    title = task[1]
    category = task[2]

    due_date = safe_parse_date(task[3])
    days_left = (due_date - datetime.now().date()).days
    is_overdue = 1 if days_left < 0 else 0

    # =========================
    # 1. RULE ENGINE SCORE
    # =========================
    rule_score, reasons = get_task_score(task)
    rule_score_norm = normalize_score(rule_score)

    # =========================
    # 2. ML MODEL SCORE
    # =========================
    ml_pred, ml_prob = predict_task(task)
    ml_score = ml_prob

    # =========================
    # 3. TRANSFORMER REASONING SCORE
    # =========================
    transformer_score = predict_transformer_score(task)

    # =========================
    # 4. MEMORY BIAS (RL + habits)
    # =========================
    most_used_category = get_memory("most_used_category")

    memory_bias = 0.0

    if most_used_category and category.lower() == most_used_category.lower():
        memory_bias += 0.1

    if is_overdue:
        memory_bias -= 0.2

    # =========================
    # 5. FINAL HYBRID SCORE (UPDATED WEIGHTS)
    # =========================
    final_score = (
            (rule_score_norm * 0.30) +
            (ml_score * 0.35) +
            (transformer_score * 0.25) +
            (memory_bias * 0.10)
        ) * 100

    # =========================
    # 6. DECISION LAYER
    # =========================
    if final_score >= 70:
        decision = "HIGH PRIORITY"
    elif final_score >= 40:
        decision = "MEDIUM PRIORITY"
    else:
        decision = "LOW PRIORITY"

    # =========================
    # 7. RETURN STRUCTURED BRAIN OUTPUT
    # =========================
    return {
    "task": title,
    "category": category,
    "due_date": str(due_date),
    "rule_score": rule_score,
    "ml_probability": ml_prob,
    "transformer_score": transformer_score,
    "memory_bias": memory_bias,
    "final_score": round(final_score, 2),
    "decision": decision,
    "reasons": reasons
}