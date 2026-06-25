# planner/task_decomposer.py

def decompose_task(task):
    title = task[1].lower()

    if "python" in title:
        return [
            "Install Python",
            "Learn syntax basics",
            "Practice small programs",
            "Build mini project"
        ]

    if "medicine" in title:
        return [
            "Check dosage instructions",
            "Set reminder time",
            "Prepare medication",
            "Take dose"
        ]

    return [f"Complete: {title}"]