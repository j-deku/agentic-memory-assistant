from conversation_memory_db import get_memory


def get_personalized_context():

    favorite_category = get_memory(
        "favorite_category"
    )

    if favorite_category:

        return (
            f"You usually focus on "
            f"{favorite_category} tasks."
        )

    return None