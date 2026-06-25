"""
agent/planner.py

Converts an NLUPipeline ParseResult (+ raw text) into a Plan.

Design rules
------------
* One logical goal  → one Plan
* Multi-step goals  (e.g. add task + schedule review) get multiple ActionSteps
* Each ActionStep names a *logical tool*, not a Python function directly —
  the Executor resolves the actual callable at runtime
* Expected outcomes are written in plain English so the Reflector can verify
  them without needing to inspect DB state itself

Logical tool names (must match keys in AgentOrchestrator._tools):
  "add_task"            – create a new task
  "list_tasks"          – retrieve all pending tasks
  "complete_task"       – mark a task done
  "delete_task"         – remove a task
  "get_overdue_tasks"   – fetch overdue items
  "productivity_score"  – compute score
  "get_recommended_tasks" – fetch suggestions
  "analyze_habits"      – habit analysis
  "dialogue"            – fall through to DialogueManager for anything else
"""

from __future__ import annotations

from brain.goal_types import Goal
from agentic.models import ActionStep, Plan


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Stateless: call build() once per user turn."""

    def build(self, result, raw_text: str) -> Plan:
        """
        Parameters
        ----------
        result   : ParseResult from NLUPipeline.parse()
        raw_text : original user input (fallback for dialogue step)

        Returns
        -------
        Plan with one or more ActionSteps
        """
        intent = result.intent if result else None
        goal   = self._describe_goal(intent, result, raw_text)
        steps  = self._steps_for_intent(intent, result, raw_text)
        return Plan(goal=goal, steps=steps)

    # ------------------------------------------------------------------ #
    # Goal description (human-readable, used in logs + reflection summary)
    # ------------------------------------------------------------------ #

    def _describe_goal(self, intent, result, raw_text: str) -> str:
        if intent == Goal.ADD_TASK:
            title = (result.title or "a new task") if result else "a new task"
            return f"Add task: '{title}'"

        if intent == Goal.LIST_TASKS:
            return "Show all pending tasks"

        if intent == Goal.COMPLETE_TASK:
            ref = (result.title or result.task_ref or "a task") if result else "a task"
            return f"Complete task: '{ref}'"

        if intent == Goal.DELETE_TASK:
            ref = (result.title or result.task_ref or "a task") if result else "a task"
            return f"Delete task: '{ref}'"

        if intent == Goal.GET_OVERDUE_TASKS:
            return "List overdue tasks"

        if intent == Goal.PRODUCTIVITY_SCORE:
            return "Get productivity score"

        if intent == Goal.GET_RECOMMENDED_TASKS:
            return "Get task recommendations"

        if intent == Goal.ANALYZE_HABITS:
            return "Analyse habit patterns"

        return f"Handle: '{raw_text[:60]}'"

    # ------------------------------------------------------------------ #
    # Step construction
    # ------------------------------------------------------------------ #

    def _steps_for_intent(
        self, intent, result, raw_text: str
    ) -> list[ActionStep]:

        # ── ADD TASK ────────────────────────────────────────────────────
        if intent == Goal.ADD_TASK:
            title    = result.title    if result else None
            category = result.category if result else None
            due      = result.due_date if result else None

            steps = []

            # Core: add the task
            steps.append(ActionStep(
                tool="add_task",
                args=dict(title=title, category=category, due_date=due),
                expected_outcome=(
                    f"Task '{title or 'new task'}' appears in task list "
                    f"with category='{category or 'personal'}'"
                    + (f" and due='{due}'" if due else "")
                ),
            ))

            # Follow-up: verify by listing (gives Reflector something to check)
            steps.append(ActionStep(
                tool="list_tasks",
                args={},
                expected_outcome=(
                    f"Task list contains '{title or 'new task'}'"
                ),
            ))

            return steps

        # ── LIST TASKS ──────────────────────────────────────────────────
        if intent == Goal.LIST_TASKS:
            return [ActionStep(
                tool="list_tasks",
                args={},
                expected_outcome="Non-empty task list returned to user",
            )]

        # ── COMPLETE TASK ───────────────────────────────────────────────
        if intent == Goal.COMPLETE_TASK:
            ref = (result.title or result.task_ref) if result else None
            return [ActionStep(
                tool="complete_task",
                args=dict(task_ref=ref, raw_text=raw_text),
                expected_outcome=(
                    f"Task matching '{ref or raw_text}' is marked complete "
                    "and no longer in pending list"
                ),
            )]

        # ── DELETE TASK ─────────────────────────────────────────────────
        if intent == Goal.DELETE_TASK:
            ref = (result.title or result.task_ref) if result else None
            return [ActionStep(
                tool="delete_task",
                args=dict(task_ref=ref, raw_text=raw_text),
                expected_outcome=(
                    f"Task matching '{ref or raw_text}' removed from task list"
                ),
            )]

        # ── ANALYTICS ───────────────────────────────────────────────────
        if intent == Goal.GET_OVERDUE_TASKS:
            return [ActionStep(
                tool="get_overdue_tasks",
                args={},
                expected_outcome="Overdue task list returned",
            )]

        if intent == Goal.PRODUCTIVITY_SCORE:
            return [ActionStep(
                tool="productivity_score",
                args={},
                expected_outcome="Numeric productivity score returned",
            )]

        if intent == Goal.GET_RECOMMENDED_TASKS:
            return [ActionStep(
                tool="get_recommended_tasks",
                args={},
                expected_outcome="Recommendation list returned",
            )]

        if intent == Goal.ANALYZE_HABITS:
            return [ActionStep(
                tool="analyze_habits",
                args={},
                expected_outcome="Habit analysis string returned",
            )]

        # ── FALLBACK: hand off to DialogueManager ───────────────────────
        return [ActionStep(
            tool="dialogue",
            args=dict(text=raw_text),
            expected_outcome="DialogueManager returns a non-empty response",
        )]