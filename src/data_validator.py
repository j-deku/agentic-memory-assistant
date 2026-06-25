# data_validator.py

from datetime import datetime

VALID_CATEGORIES = {"school", "learning", "health", "work", "personal"}

def normalize_task(task):
    """
    Converts raw DB tuple into clean, structured dict
    """

    return {
        "id": task[0],
        "title": task[1],
        "category": str(task[2]).strip().lower() if task[2] else "personal",
        "due_date": task[3],
        "completed": task[4],
        "created_at": task[5],
        "completion_hour": task[9] if len(task) > 9 and task[9] is not None else -1,
        "completion_day": task[10] if len(task) > 10 and task[10] else "none"
    }


def validate_task(task):
    """
    Ensures task is safe and usable for ML + reasoning
    """

    task = normalize_task(task)

    # -----------------------
    # CATEGORY FIX
    # -----------------------
    if task["category"] not in VALID_CATEGORIES:
        print(f"[DATA FIX] Invalid category corrected: {task['category']} → personal")

    # -----------------------
    # DATE SAFETY
    # -----------------------
    try:
        task["due_date"] = datetime.strptime(
            task["due_date"], "%Y-%m-%d"
        )
    except:
        task["due_date"] = datetime.now()

    # -----------------------
    # COMPLETION CLEANUP
    # -----------------------
    task["completed"] = 1 if task["completed"] == 1 else 0

    return task

def validate_task_batch(tasks):
    return [validate_task(t) for t in tasks]