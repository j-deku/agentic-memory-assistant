# context_handler.py

from datetime import datetime, timedelta

from tasks import add_task

from conversation_memory import (
    get_pending_action,
    set_pending_action,
    clear_pending_action,
    set_memory,
    get_memory
)


def handle_context(user_input):

    pending = get_pending_action()

    # =========================
    # STEP 1
    # User starts reminder flow
    # =========================

    if pending is None:

        text = user_input.lower()

        if "remind me tomorrow" in text:
            set_pending_action("waiting_for_task")

            due_date = (
                datetime.now() +
                timedelta(days=1)
            ).strftime("%Y-%m-%d")

            set_memory("due_date", due_date)

            return "What should I remind you about?"

        return None

    # =========================
    # STEP 2
    # Waiting for task title
    # =========================

    if pending == "waiting_for_task":

        title = user_input

        due_date = get_memory("due_date")

        add_task(
            title,
            "personal",
            due_date
        )

        clear_pending_action()

        return (
            f"✅ Reminder created: {title}\n"
            f"Due: {due_date}"
        )

    return None