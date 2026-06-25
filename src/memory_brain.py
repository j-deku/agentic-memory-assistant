from conversation_memory_db import (
    save_memory,
    get_memory,
    get_all_memories
)


def learn_from_text(text):

    text = text.lower()

    if "my favorite category is" in text:

        category = (
            text.replace(
                "my favorite category is",
                ""
            )
            .strip()
        )

        save_memory(
            "favorite_category",
            category
        )

        return (
            f"I'll remember that your "
            f"favorite category is {category}."
        )

    return None


def answer_memory_question(text):

    text = text.lower()

    if "what do you know about me" in text:

        memories = get_all_memories()

        if not memories:
            return "I haven't learned much about you yet."

        response = "Here's what I remember:\n"

        for key, value in memories:
            response += f"- {key}: {value}\n"

        return response

    return None 