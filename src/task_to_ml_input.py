def task_to_ml_input(task):
    return {
        "category": task["category"],
        "created_at": task["created_at"],
        "due_date": task["due_date"],
        "completion_hour": task["completion_hour"],
        "completion_day": task["completion_day"]
    }