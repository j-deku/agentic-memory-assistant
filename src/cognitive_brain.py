from tasks import list_tasks, get_memory, update_memory_from_tasks
from datetime import datetime
from learning_brain import get_weight
from behavior_profile import build_behavior_profile
from predict_model import predict_task_completion

# =========================
# 1. MEMORY ENGINE
# =========================

def load_user_memory():
    return {
        "most_used_category": get_memory("most_used_category"),
        "completion_rate": get_memory("completion_rate"),
        "total_tasks": get_memory("total_tasks"),
    }


# =========================
# 2. CONTEXT BUILDER
# =========================

def build_context():
    tasks = list_tasks()
    memory = load_user_memory()

    today = datetime.now().date()

    enriched_tasks = []

    for t in tasks:
        due_date = datetime.strptime(t[3], "%Y-%m-%d").date()
        days_left = (due_date - today).days

        enriched_tasks.append({
            "id": t[0],
            "title": t[1],
            "category": t[2],
            "due_date": t[3],
            "completed": t[4],
            "days_left": days_left
        })

    return enriched_tasks, memory


# =========================
# 3. REASONING ENGINE
# =========================

def reason(tasks, memory):

    reasoning_output = []

    # Build learned behavior profile once
    profile = build_behavior_profile()

    current_hour = datetime.now().hour

    for t in tasks:

        score = 0
        reasons = []

        # =========================
        # 1. URGENCY LAYER
        # =========================

        if t["days_left"] < 0:
            score += 50
            reasons.append("Overdue task")

        elif t["days_left"] <= 2:
            score += 30
            reasons.append("Due soon")

        # =========================
        # 2. COMPLETION STATE
        # =========================

        if t["completed"] == 1:
            score -= 40
            reasons.append("Already completed")

        # =========================
        # 3. ML PREDICTION LAYER
        # =========================

        try:

            prob = predict_task_completion({
                "category": t["category"],
                "created_at": datetime.now(),
                "due_date": t["due_date"],
                "completion_hour": t.get("completion_hour", -1),
                "completion_day": t.get("completion_day", "none")
            })

            score += prob * 100

            reasons.append(
                f"ML completion probability: {prob:.2f}"
            )

        except Exception:
            reasons.append(
                "ML prediction unavailable"
            )

        # =========================
        # 4. BEHAVIOR PROFILE LAYER
        # =========================

        if profile:

            # Category success history
            category_perf = profile.get(
                "category_performance",
                {}
            )

            if t["category"] in category_perf:

                performance = category_perf[t["category"]]

                score += performance * 20

                reasons.append(
                    f"Learned category pattern ({performance:.2f})"
                )

            # Favorite category
            if (
                profile.get("favorite_category")
                == t["category"]
            ):

                score += 10

                reasons.append(
                    "Favorite category"
                )

            # Productive hour matching
            if "best_hour" in profile:

                if current_hour == profile["best_hour"]:

                    score += 15

                    reasons.append(
                        "Matches productive hour"
                    )

        # =========================
        # 5. MEMORY LAYER
        # =========================

        if (
            memory.get("most_used_category")
            == t["category"]
        ):

            score += 10

            reasons.append(
                "Frequently selected category"
            )

        completion_rate = memory.get(
            "completion_rate"
        )

        if (
            completion_rate is not None
            and completion_rate < 50
            and t["category"] == "personal"
        ):

            score += 15

            reasons.append(
                "Historically difficult category"
            )

        # =========================
        # FINAL OUTPUT
        # =========================

        reasoning_output.append({
            "task": t,
            "score": round(score, 2),
            "reasons": reasons
        })

    return reasoning_output

# =========================
# 4. FINAL DECISION ENGINE
# =========================

def cognitive_brain():
    tasks, memory = build_context()
    reasoning = reason(tasks, memory)

    # sort by score
    reasoning.sort(key=lambda x: x["score"], reverse=True)

    return reasoning[:5]