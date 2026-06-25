# conversation_engine.py

from datetime import datetime, timedelta
from tasks import add_task


def process_message(text):

    text = text.lower()

    # =====================
    # ADD TASK
    # =====================

    if "add a task" in text:

        if "tomorrow" in text:
            due_date = (
                datetime.now() + timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            due_date = datetime.now().strftime("%Y-%m-%d")

        task_name = (
            text.replace("add a task", "")
                .replace("tomorrow", "")
                .strip()
        )

        add_task(
            task_name,
            "personal",
            due_date
        )

        return (
            f"✅ Added task: {task_name}\n"
            f"Due: {due_date}"
        )

    return None