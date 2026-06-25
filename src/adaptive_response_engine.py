from conversation_memory_db import get_memory


def personalize_response(text):

    favorite = get_memory(
        "favorite_category"
    )

    if favorite:

        return (
            f"{text}\n\n"
            f"Based on your history, "
            f"you often prioritize "
            f"{favorite} tasks."
        )

    return text