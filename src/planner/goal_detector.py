# planner/goal_detector.py

def detect_goal(task):
    title = task[1].lower()
    category = task[2].lower()

    if category in ["health"]:
        return "health_management"

    if "learn" in title or "study" in title:
        return "skill_development"

    if "visit" in title or "meet" in title:
        return "personal_commitment"

    if "work" in category:
        return "professional_task"

    return "general_task"