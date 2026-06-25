from datetime import datetime

def task_to_features(task):
    title = task[1]
    category = str(task[2]).lower()

    due_date = datetime.strptime(task[3], "%Y-%m-%d").date()
    days_left = (due_date - datetime.now().date()).days
    is_overdue = 1 if days_left < 0 else 0

    return {
        "task_length": len(title),
        "category": category,
        "days_left": days_left,
        "is_overdue": is_overdue
    }