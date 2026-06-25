from tasks import get_memory


def memory_greeting():
    most_used = get_memory("most_used_category")

    if most_used:
        print(
            f"You've been focusing heavily on "
            f"{most_used} tasks lately."
        )