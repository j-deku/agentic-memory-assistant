# semantic_parser.py

import re
from datetime import datetime, timedelta


# =========================
# DATE PARSER
# =========================
def parse_due_date(text):
    text = text.lower()

    today = datetime.now().date()

    if "today" in text:
        return str(today)

    if "tomorrow" in text:
        return str(today + timedelta(days=1))

    return None


# =========================
# SAFETY FILTER (IMPORTANT FIX)
# =========================
BLOCKED_TASK_PHRASES = [
    "what should i do",
    "how can i help",
    "show my tasks",
    "view tasks",
    "hello",
    "hi",
    "good morning",
    "good evening",
    "good afternoon"
]


# =========================
# MAIN SEMANTIC PARSER
# =========================
def semantic_action_parser(user_input):

    text = user_input.lower().strip()

    # =========================
    # 🚫 BLOCK CHAT FROM BECOMING TASKS
    # =========================
    if any(phrase in text for phrase in BLOCKED_TASK_PHRASES):
        return {"action": "chat"}

    # =========================
    # ADD TASK DETECTION (NO KEYWORDS ONLY)
    # =========================
    task_indicators = [
        "call", "buy", "eat", "drink", "go", "visit",
        "study", "learn", "do", "finish", "write"
    ]

    if any(word in text.split() for word in task_indicators):

        cleaned_title = re.sub(
            r"\b(today|tomorrow|please|i need to|i want to|to)\b",
            "",
            text
        ).strip()

        return {
            "action": "add_task",
            "title": cleaned_title,
            "due_date": parse_due_date(text),
            "category": "personal"
        }

    # =========================
    # VIEW TASKS
    # =========================
    if "my tasks" in text or "show tasks" in text:
        return {"action": "view_tasks"}

    # =========================
    # COMPLETE TASK
    # =========================
    if "complete" in text or "done" in text:
        return {"action": "complete_task"}

    # =========================
    # FALLBACK CHAT MODE
    # =========================
    return {"action": "chat"}