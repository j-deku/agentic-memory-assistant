from typing import Any, Dict, List, Optional
from datetime import datetime


class ResponseGenerator:
    """
    Converts execution results into natural language responses.

    This is the "voice layer" of the assistant.
    """

    def __init__(self, user_name: str = ""):
        self.user_name = user_name

    # ---------------------------------------------------
    # MAIN ENTRY
    # ---------------------------------------------------

    def generate(self, plan_goal, execution_result: Any, context: Dict[str, Any] = None) -> str:
        context = context or {}

        if plan_goal == "ADD_TASK":
            return self._format_add_task(execution_result, context)

        if plan_goal == "COMPLETE_TASK":
            return self._format_complete_task(execution_result, context)

        if plan_goal == "DELETE_TASK":
            return self._format_delete_task(execution_result, context)

        if plan_goal == "VIEW_TASKS":
            return self._format_view_tasks(execution_result)

        if plan_goal == "CHAT":
            return self._format_chat(execution_result)

        return "I handled that request."

    # ---------------------------------------------------
    # ADD TASK
    # ---------------------------------------------------

    def _format_add_task(self, result: Any, context: Dict[str, Any]) -> str:
        title = context.get("title", "your task")
        due = context.get("due_date")

        if due:
            return f"Got it — I’ve added \"{title}\" and set it for {due}."
        return f"Done — I’ve added \"{title}\" to your task list."

    # ---------------------------------------------------
    # COMPLETE TASK
    # ---------------------------------------------------

    def _format_complete_task(self, result: Any, context: Dict[str, Any]) -> str:
        task_name = context.get("task_title") or context.get("task_ref", "the task")

        if isinstance(result, dict) and result.get("already_done"):
            return f"\"{task_name}\" was already marked as completed."

        return f"Nice — I’ve marked \"{task_name}\" as done. ✓"

    # ---------------------------------------------------
    # DELETE TASK
    # ---------------------------------------------------

    def _format_delete_task(self, result: Any, context: Dict[str, Any]) -> str:
        task_name = context.get("task_title") or context.get("task_ref", "that task")

        if context.get("confirmed") is False:
            return f"Okay — I won’t delete \"{task_name}\"."

        return f"Deleted \"{task_name}\" successfully. ✓"

    # ---------------------------------------------------
    # VIEW TASKS
    # ---------------------------------------------------

    def _format_view_tasks(self, tasks: List[Any]) -> str:
        if not tasks:
            return "You don’t have any tasks right now."

        pending = [t for t in tasks if not self._is_completed(t)]

        if not pending:
            return "You’ve completed everything — nice work!"

        lines = [f"You have {len(pending)} pending tasks:\n"]

        for i, t in enumerate(pending[:10], 1):
            title = self._get(t, "title", 1)
            due = self._get(t, "due_date", 3)
            lines.append(f"{i}. {title} (due {due or 'no date'})")

        return "\n".join(lines)

    # ---------------------------------------------------
    # CHAT RESPONSE
    # ---------------------------------------------------

    def _format_chat(self, text: str) -> str:
        greetings = ["hi", "hello", "hey"]

        if any(g in text.lower() for g in greetings):
            return f"Hey {self.user_name or ''}! What would you like to do today?"

        return text

    # ---------------------------------------------------
    # HELPERS
    # ---------------------------------------------------

    def _get(self, obj, key: str, index: int):
        if isinstance(obj, dict):
            return obj.get(key)
        try:
            return obj[index]
        except Exception:
            return None

    def _is_completed(self, task):
        if isinstance(task, dict):
            return task.get("completed", False)
        try:
            return bool(task[4])
        except Exception:
            return False