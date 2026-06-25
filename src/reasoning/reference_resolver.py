def resolve_reference(
    text,
    conversation_state
):

    lower = text.lower()

    if "that" in lower:
        return conversation_state.last_task

    if "it" in lower:
        return conversation_state.last_task

    if "this" in lower:
        return conversation_state.last_task

    if "first" in lower:

        tasks = (
            conversation_state
            .last_recommended_tasks
        )

        if tasks:
            return tasks[0]

    if "second" in lower:

        tasks = (
            conversation_state
            .last_recommended_tasks
        )

        if len(tasks) > 1:
            return tasks[1]

    if "third" in lower:

        tasks = (
            conversation_state
            .last_recommended_tasks
        )

        if len(tasks) > 2:
            return tasks[2]

    return None