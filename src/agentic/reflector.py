"""
agent/reflector.py

Post-execution analysis.

reflect() inspects the Plan after the Executor has run all pending steps and
returns a ReflectionResult that tells the Orchestrator:

  1. Did everything succeed?                        (success bool)
  2. How well did it go?                            (score 0.0–1.0)
  3. What is still unmet?                           (unmet_goals list)
  4. What should the user hear?                     (summary str)
  5. Should we replan?                              (should_replan bool)

Verification strategy
---------------------
For ADD_TASK the Reflector cross-checks the task list result (step index 1)
to confirm the new title actually appears.  For all other intents a SUCCESS
status on the step is sufficient evidence.

Scoring
-------
  score = succeeded_steps / (total_steps - skipped_steps)

A score of 1.0 means every meaningful step succeeded.
"""

from __future__ import annotations

from agentic.models import (
    Plan,
    PlanStatus,
    ReflectionResult,
    StepStatus,
)


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

class Reflector:

    def reflect(
        self,
        plan: Plan,
        max_replan_attempts: int,
    ) -> ReflectionResult:
        """
        Analyse plan state after one execution pass.

        Parameters
        ----------
        plan                : mutated in-place by Executor (statuses set)
        max_replan_attempts : ceiling from Orchestrator constant
        """
        total    = len(plan.steps)
        skipped  = sum(1 for s in plan.steps if s.status == StepStatus.SKIPPED)
        done     = sum(1 for s in plan.steps if s.status == StepStatus.SUCCESS)
        failed   = plan.failed_steps

        effective_total = total - skipped
        score = (done / effective_total) if effective_total > 0 else 1.0

        unmet_goals  = self._collect_unmet(plan)
        success      = len(failed) == 0 and len(unmet_goals) == 0
        should_replan = (
            not success
            and bool(unmet_goals or failed)
            and plan.replan_count < max_replan_attempts
        )

        summary = self._build_summary(plan, score, unmet_goals, success)

        return ReflectionResult(
            success=success,
            score=round(score, 2),
            unmet_goals=unmet_goals,
            summary=summary,
            should_replan=should_replan,
        )

    # ------------------------------------------------------------------ #
    # Unmet-goal detection
    # ------------------------------------------------------------------ #

    def _collect_unmet(self, plan: Plan) -> list[str]:
        unmet = []

        for step in plan.steps:
            if step.status == StepStatus.FAILED:
                unmet.append(
                    f"Step '{step.tool}' failed"
                    + (f": {step.error}" if step.error else "")
                )
                continue

            if step.status != StepStatus.SUCCESS:
                continue

            # Extra semantic check for add_task verification step
            if (
                step.tool == "list_tasks"
                and step.result is not None
            ):
                # Find the preceding add_task step to get the expected title
                add_step = self._preceding_add_step(plan, step)
                if add_step:
                    expected_title = (
                        add_step.args.get("title") or ""
                    ).lower().strip()
                    tasks = step.result if isinstance(step.result, list) else []
                    found = any(
                        expected_title in self._task_title(t).lower()
                        for t in tasks
                    ) if expected_title else True   # can't verify without title

                    if not found:
                        unmet.append(
                            f"Task '{expected_title}' not found in list "
                            "after add — possible save failure"
                        )

        return unmet

    def _preceding_add_step(self, plan: Plan, list_step):
        """Return the add_task step immediately before list_step, if any."""
        for i, s in enumerate(plan.steps):
            if s is list_step and i > 0:
                prev = plan.steps[i - 1]
                if prev.tool == "add_task":
                    return prev
        return None

    @staticmethod
    def _task_title(task) -> str:
        if isinstance(task, (list, tuple)):
            return str(task[1]) if len(task) > 1 else ""
        if isinstance(task, dict):
            return task.get("title", "")
        return str(task)

    # ------------------------------------------------------------------ #
    # Summary builder
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        plan: Plan,
        score: float,
        unmet_goals: list[str],
        success: bool,
    ) -> str:
        succeeded = plan.succeeded_steps
        failed    = plan.failed_steps

        parts: list[str] = []

        if success:
            parts.append(f"✅ Goal completed: {plan.goal}")
            if len(succeeded) > 1:
                parts.append(
                    f"   {len(succeeded)} steps executed successfully."
                )
        else:
            pct = int(score * 100)
            parts.append(
                f"⚠️  Partial completion ({pct}%): {plan.goal}"
            )
            if succeeded:
                titles = ", ".join(s.tool for s in succeeded)
                parts.append(f"   ✓ Succeeded: {titles}")
            if failed:
                titles = ", ".join(s.tool for s in failed)
                parts.append(f"   ✗ Failed:    {titles}")
            if unmet_goals:
                parts.append("   Unmet goals:")
                for g in unmet_goals:
                    parts.append(f"     • {g}")

        if plan.replan_count > 0:
            parts.append(f"   (replanned {plan.replan_count} time(s))")

        return "\n".join(parts)