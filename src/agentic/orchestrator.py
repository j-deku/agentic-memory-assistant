"""
agent/orchestrator.py

AgentOrchestrator — the top-level agentic loop.

Flow per user turn
------------------

  run(user_input)
    │
    ├─ GOAL    : NLUPipeline.parse() → intent + slots
    │
    ├─ PLAN    : Planner.build()     → Plan(steps=[ActionStep, ...])
    │
    ├─ EXECUTE : _execute_pending()  → runs each PENDING step in order
    │              └─ each step calls the matching tool in self._tools
    │                 and marks the step SUCCESS / FAILED
    │
    ├─ REFLECT : Reflector.reflect() → ReflectionResult
    │              ├─ verifies outcomes
    │              ├─ scores the plan  (0.0–1.0)
    │              ├─ logs via log_event
    │              └─ returns should_replan flag
    │
    └─ REPLAN  : if should_replan → _replan() → mutate Plan → back to EXECUTE
                 (max MAX_REPLAN_ATTEMPTS times, then graceful degradation)

Tool resolution
---------------
Logical tool names (set by Planner) are resolved to callables via
self._tools dict.  DialogueManager is wired as the "dialogue" tool so
slot-filling flows still work unchanged.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from agentic.models import (
    ActionStep,
    ExecutionResult,
    Plan,
    PlanStatus,
    StepStatus,
)
from agentic.planner import Planner
from agentic.reflector import Reflector

logger = logging.getLogger(__name__)

MAX_REPLAN_ATTEMPTS = 3   # 2–3 balanced; change here only


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Parameters
    ----------
    nlu          : NLUPipeline instance (shared with DialogueManager)
    dialogue_mgr : DialogueManager instance (handles slot-filling + responses)
    task_fns     : dict of raw task callables from main.py (_task_fns)
    log_event    : callable(event_type, data) — existing logging hook
    """

    def __init__(
        self,
        nlu,
        dialogue_mgr,
        task_fns: dict,
        log_event: Callable | None = None,
    ):
        self.nlu      = nlu
        self.dm       = dialogue_mgr
        self.planner  = Planner()
        self.reflector = Reflector()
        self._log     = log_event or (lambda *a, **k: None)

        # Map logical tool names → callables
        self._tools: dict[str, Callable] = {
            "add_task":               task_fns.get("add_task"),
            "list_tasks":             task_fns.get("list_tasks"),
            "complete_task":          self._tool_complete,
            "delete_task":            self._tool_delete,
            "get_overdue_tasks":      task_fns.get("get_overdue_tasks"),
            "productivity_score":     task_fns.get("productivity_score"),
            "get_recommended_tasks":  task_fns.get("get_recommended_tasks"),
            "analyze_habits":         task_fns.get("analyze_habits"),
            "dialogue":               self._tool_dialogue,
        }

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> str:
        """
        Process one user turn through the full agentic loop.
        Always returns a string suitable for printing to the user.
        """
        # ── GOAL ────────────────────────────────────────────────────────
        result = self.nlu.parse(user_input)
        if self.dm.state.awaiting_slot:
            return self.dm.process(user_input)

        # ── PLAN ────────────────────────────────────────────────────────
        plan = self.planner.build(result, user_input)

        self._log("agent_plan_created", {
            "goal":       plan.goal,
            "step_count": len(plan.steps),
            "timestamp":  plan.created_at.isoformat(),
        })

        # ── EXECUTE → REFLECT → (REPLAN) loop ───────────────────────────
        final_response   = ""
        reflection       = None

        while True:
            responses = self._execute_pending(plan)

            reflection = self.reflector.reflect(plan, MAX_REPLAN_ATTEMPTS)

            self._log("agent_reflection", {
                "goal":          plan.goal,
                "score":         reflection.score,
                "success":       reflection.success,
                "replan_count":  plan.replan_count,
                "unmet":         reflection.unmet_goals,
                "timestamp":     datetime.now().isoformat(),
            })

            logger.debug(
                "[Orchestrator] %s | score=%.2f | replan=%d/%d",
                "OK" if reflection.success else "PARTIAL",
                reflection.score,
                plan.replan_count,
                MAX_REPLAN_ATTEMPTS,
            )

            if reflection.success or not reflection.should_replan:
                # Collect the last meaningful response from execution
                final_response = self._best_response(responses, plan)
                break

            # ── REPLAN ──────────────────────────────────────────────────
            plan = self._replan(plan, reflection, user_input)

        # ── SUMMARISE ───────────────────────────────────────────────────
        plan.status = (
            PlanStatus.COMPLETE
            if reflection and reflection.success
            else (
                PlanStatus.PARTIAL
                if plan.succeeded_steps
                else PlanStatus.ABANDONED
            )
        )

        self._log("agent_plan_closed", {
            "goal":    plan.goal,
            "status":  plan.status.name,
            "score":   reflection.score if reflection else 0,
        })

        # Append reflection summary in debug / verbose mode if partial
        if reflection and not reflection.success and reflection.summary:
            logger.info("\n%s", reflection.summary)

        return final_response or "I wasn't able to complete that. Please try again."

    # ------------------------------------------------------------------ #
    # Execution pass
    # ------------------------------------------------------------------ #

    def _execute_pending(self, plan: Plan) -> list[ExecutionResult]:
        """Run every PENDING step; update step status in-place."""
        results = []

        for step in plan.pending_steps:
            step.attempts += 1
            er = self._run_step(step)
            results.append(er)

            if not er.ok:
                # Non-fatal: log and continue — Reflector decides whether to replan
                logger.warning(
                    "[Executor] Step '%s' failed (attempt %d): %s",
                    step.tool, step.attempts, step.error,
                )

        return results

    def _run_step(self, step: ActionStep) -> ExecutionResult:
        tool_fn = self._tools.get(step.tool)

        if tool_fn is None:
            step.status = StepStatus.FAILED
            step.error  = f"Unknown tool: '{step.tool}'"
            return ExecutionResult(step=step, response="", ok=False)

        try:
            raw = tool_fn(**step.args)
            step.result = raw
            step.status = StepStatus.SUCCESS
            response    = str(raw) if raw is not None else ""
            return ExecutionResult(step=step, response=response, ok=True)

        except Exception as exc:  # noqa: BLE001
            step.status = StepStatus.FAILED
            step.error  = str(exc)
            return ExecutionResult(step=step, response="", ok=False)

    # ------------------------------------------------------------------ #
    # Replan
    # ------------------------------------------------------------------ #

    def _replan(
        self, plan: Plan, reflection, user_input: str
    ) -> Plan:
        """
        Mutate the plan to retry only failed/unmet steps.

        Strategy
        --------
        1. Reset FAILED steps to PENDING (they will re-run next pass).
        2. If the failure looks like a missing title (slot gap), inject a
           dialogue step so DialogueManager can collect it.
        3. Increment replan_count.
        """
        plan.replan_count += 1

        logger.info(
            "[Replanner] Replan #%d for goal: %s",
            plan.replan_count, plan.goal,
        )

        for step in plan.failed_steps:
            missing_title = (
                step.tool == "add_task"
                and not step.args.get("title")
            )

            if missing_title:
                # Replace with a dialogue step that collects the slot
                step.tool   = "dialogue"
                step.args   = {"text": user_input}
                step.expected_outcome = "DialogueManager collects missing title slot"

            step.status  = StepStatus.PENDING
            step.error   = None

        # Re-add list_tasks verification if we reset an add_task step
        tools_in_plan = {s.tool for s in plan.steps}
        if "add_task" in tools_in_plan and "list_tasks" not in tools_in_plan:
            plan.steps.append(ActionStep(
                tool="list_tasks",
                args={},
                expected_outcome="Verify task list after retry",
            ))

        self._log("agent_replan", {
            "replan_count":  plan.replan_count,
            "goal":          plan.goal,
            "unmet":         reflection.unmet_goals,
            "timestamp":     datetime.now().isoformat(),
        })

        return plan

    # ------------------------------------------------------------------ #
    # Tool adapters
    # ------------------------------------------------------------------ #

    def _tool_dialogue(self, text: str) -> str:
        """Route to DialogueManager for slot-fill flows and fallbacks."""
        return self.dm.process(text)

    def _tool_complete(self, task_ref=None, raw_text: str = "") -> str:
        """
        Resolve and complete a task by delegating to DialogueManager's
        internal helper so fuzzy matching stays in one place.
        """
        search = task_ref or raw_text
        match  = self.dm._find_and_complete_by_name(search)
        if match:
            return match
        raise ValueError(f"No task found matching '{search}'")

    def _tool_delete(self, task_ref=None, raw_text: str = "") -> str:
        search = task_ref or raw_text
        match  = self.dm._find_and_delete_by_name(search)
        if match:
            return match
        raise ValueError(f"No task found matching '{search}'")

    # ------------------------------------------------------------------ #
    # Response selection
    # ------------------------------------------------------------------ #

    """
    orchestrator_patch.py

    Replace the _best_response method in your existing orchestrator.py
    Also patches _execute_pending to format responses properly.

    INSTRUCTIONS:
    Copy the _best_response and _format_task_list methods below
    into your src/agentic/orchestrator.py, replacing the existing
    _best_response method.
    """

    # ---------------------------------------------------------------------------
    # Paste these two methods into AgentOrchestrator class in orchestrator.py
    # ---------------------------------------------------------------------------

    def _format_task_list(self, tasks) -> str:
        """Format raw task tuples/dicts into readable text."""
        if not tasks:
            return "You have no tasks right now."
        lines = [f"Here are your {len(tasks)} pending task{'s' if len(tasks) != 1 else ''}:"]
        for t in tasks[:8]:
            if isinstance(t, (list, tuple)):
                title    = t[1] if len(t) > 1 else "Unknown"
                due      = t[3] if len(t) > 3 else None
                category = t[2] if len(t) > 2 else None
            else:
                title    = t.get("title", "Unknown")
                due      = t.get("due_date")
                category = t.get("category")
            line = f"  • {title.capitalize()}"
            if due and str(due) not in ("None", ""):
                line += f" — due {due}"
            if category and str(category) not in ("None", ""):
                line += f" ({category})"
            lines.append(line)
        if len(tasks) > 8:
            lines.append(f"  ...and {len(tasks) - 8} more.")
        return "\n".join(lines)


    def _format_overdue(self, tasks) -> str:
        """Format raw overdue task tuples into readable text."""
        if not tasks:
            return "Great news — you have no overdue tasks!"
        lines = [f"You have {len(tasks)} overdue task{'s' if len(tasks) != 1 else ''} that need attention:"]
        for t in tasks:
            if isinstance(t, (list, tuple)):
                title = t[1] if len(t) > 1 else "Unknown"
                due   = t[3] if len(t) > 3 else None
            else:
                title = t.get("title", "Unknown")
                due   = t.get("due_date")
            line = f"  • {title.capitalize()}"
            if due and str(due) not in ("None", ""):
                line += f" — was due {due}"
            lines.append(line)
        lines.append("\nWould you like to complete or reschedule any of these?")
        return "\n".join(lines)


    def _format_recommended(self, tasks) -> str:
        """Format raw recommended task tuples into readable text."""
        if not tasks:
            return "You're all caught up — no recommendations right now."
        lines = ["Here's what I recommend you focus on next:"]
        for t in tasks[:5]:
            if isinstance(t, (list, tuple)):
                title    = t[1] if len(t) > 1 else "Unknown"
                category = t[2] if len(t) > 2 else None
            else:
                title    = t.get("title", "Unknown")
                category = t.get("category")
            line = f"  • {title.capitalize()}"
            if category and str(category) not in ("None", ""):
                line += f" ({category})"
            lines.append(line)
        return "\n".join(lines)


    def _best_response(self, results, plan) -> str:
        """
        Pick the most user-facing response from the execution results.

        Priority:
        1. dialogue step response  (already formatted for the user)
        2. add_task confirmation
        3. complete / delete confirmation
        4. analytics (productivity, overdue, recommended, habits)
        5. list_tasks (formatted)
        6. empty string (caller handles)
        """
        # 1. Dialogue step — already user-facing
        dialogue_resp = next(
            (r.response for r in results if r.step.tool == "dialogue" and r.ok),
            None,
        )
        if dialogue_resp:
            return dialogue_resp

        # 2. Add task — return confirmation, NOT the subsequent list_tasks
        for step in plan.succeeded_steps:
            if step.tool == "add_task" and isinstance(step.result, str):
                return step.result

        # 3. Complete / delete
        for step in reversed(plan.succeeded_steps):
            if step.tool in ("complete_task", "delete_task") and isinstance(step.result, str):
                return step.result

        # 4. Analytics — format properly instead of raw str()
        for step in reversed(plan.succeeded_steps):
            if step.tool == "get_overdue_tasks" and step.result is not None:
                return self._format_overdue(step.result)
            if step.tool == "get_recommended_tasks" and step.result is not None:
                return self._format_recommended(step.result)
            if step.tool == "productivity_score" and step.result is not None:
                return str(step.result)
            if step.tool == "analyze_habits" and step.result is not None:
                return str(step.result)

        # 5. List tasks — formatted
        for step in reversed(plan.succeeded_steps):
            if step.tool == "list_tasks" and step.result is not None:
                return self._format_task_list(step.result)

        return ""