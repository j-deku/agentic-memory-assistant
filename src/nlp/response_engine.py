"""
response_engine.py  —  Fixed version

Changes:
  - overdue_tasks() now formats tuples into readable text
  - recommended_tasks() now formats tuples into readable text  
  - All responses are professional and natural
  - task_list() shows due dates when available
"""

import random
from datetime import datetime


def _format_task(t) -> str:
    """Format a raw task tuple into a readable string."""
    if isinstance(t, dict):
        title    = t.get("title", "Unknown")
        due      = t.get("due_date", None)
        category = t.get("category", None)
    else:
        # Tuple format: (id, title, category, due_date, completed, ...)
        title    = t[1] if len(t) > 1 else "Unknown"
        category = t[2] if len(t) > 2 else None
        due      = t[3] if len(t) > 3 else None

    title = title.capitalize() if title else "Unknown"

    parts = [f"• {title}"]
    if due and due not in ("None", None, ""):
        parts[0] += f" — due {due}"
    if category and category not in ("None", None, ""):
        parts[0] += f" ({category})"

    return parts[0]


class ResponseEngine:
    def __init__(self, user_name: str = ""):
        self.user_name = user_name

    def task_added(self, title, due, category):
        name = self.user_name or "there"
        due_str = f" for {due}" if due else ""
        cat_str = f" under {category}" if category else ""
        return random.choice([
            f"Done! I've added '{title}'{due_str}{cat_str} to your list.",
            f"Got it, {name} — '{title}' has been saved{due_str}.",
            f"'{title}' has been added successfully{due_str}.",
        ])

    def task_list(self, tasks):
        if not tasks:
            return "Your task list is clear. Nothing pending right now."

        lines = [f"Here are your {len(tasks)} pending task{'s' if len(tasks) != 1 else ''}:"]
        for t in tasks[:8]:
            lines.append(_format_task(t))
        if len(tasks) > 8:
            lines.append(f"  ...and {len(tasks) - 8} more.")
        return "\n".join(lines)

    def task_completed(self, title):
        return random.choice([
            f"Great work! '{title}' has been marked as complete ✓",
            f"Done — '{title}' is complete. Keep it up!",
            f"'{title}' checked off your list ✓",
        ])

    def task_deleted(self, title):
        return f"Done. '{title}' has been removed from your list ✓"

    def recommended_tasks(self, tasks):
        if not tasks:
            return "You're all caught up — no recommendations right now."

        lines = [f"Here's what I recommend you focus on next:"]
        for t in tasks[:5]:
            lines.append(_format_task(t))
        return "\n".join(lines)

    def overdue_tasks(self, tasks):
        if not tasks:
            return f"Great news — you have no overdue tasks!"

        lines = [f"You have {len(tasks)} overdue task{'s' if len(tasks) != 1 else ''} that need attention:"]
        for t in tasks:
            lines.append(_format_task(t))
        lines.append("\nWould you like to complete or reschedule any of these?")
        return "\n".join(lines)

    def check_task(self, title):
        if not title:
            return "I couldn't find that task in your list."
        return f"Yes — '{title}' is in your list. Have you completed it already?"