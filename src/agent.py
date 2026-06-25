from tasks import (
    list_tasks,
    get_recommended_tasks,
    get_task_score,
    add_task,
    complete_task
)

from ai_brain import hybrid_decision
from datetime import datetime


def get_today_context():
    tasks = list_tasks()

    if not tasks:
        return None

    scored = []

    for task in tasks:
        result = hybrid_decision(task)
        scored.append(result)

    # sort by priority
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    return scored


def autonomous_plan():
    tasks = get_today_context()

    if not tasks:
        return ["No tasks available. Suggest creating a new habit task."]

    plan = []

    high_priority = [t for t in tasks if t["final_score"] >= 70]
    medium_priority = [t for t in tasks if 40 <= t["final_score"] < 70]

    # 1. Focus block (top task)
    if high_priority:
        top = high_priority[0]
        plan.append(f"Focus on: {top['task']} ({top['category']})")

    # 2. Secondary tasks
    for t in high_priority[1:3]:
        plan.append(f"Next: {t['task']}")

    # 3. Optional tasks
    for t in medium_priority[:2]:
        plan.append(f"Optional: {t['task']}")

    return plan


def autonomous_actions():
    """
    This is where the agent actually "acts"
    (SAFE ACTIONS ONLY)
    """

    tasks = get_today_context()

    if not tasks:
        return "No actions needed."

    actions = []

    # Example autonomy rules
    for t in tasks:

        # If ML + rules say very low chance → suggest skipping focus
        if t["final_score"] < 30:
            actions.append(f"Consider postponing: {t['task']}")

        # If overdue + high priority → escalate
        if "Overdue" in t["decision"]:
            actions.append(f"Urgent attention required: {t['task']}")

    if not actions:
        actions.append("System stable. Continue normal workflow.")

    return actions