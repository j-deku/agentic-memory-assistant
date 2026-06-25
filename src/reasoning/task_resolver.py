from rapidfuzz import process

def resolve_task_reference(
    user_text,
    tasks,
    threshold=70
):

    if not tasks:
        return None

    names = [
        task["title"]
        for task in tasks
    ]

    result = process.extractOne(
        user_text,
        names
    )

    if not result:
        return None

    name, score, _ = result

    if score >= threshold:
        return name

    return None
