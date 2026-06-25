from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from brain.goal_types import Goal


# ---------------------------------------------------
# PLAN STRUCTURES
# ---------------------------------------------------

@dataclass
class PlanStep:
    action: str
    slots: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Plan:
    goal: Goal
    steps: List[PlanStep] = field(default_factory=list)
    confidence: float = 1.0
    requires_confirmation: bool = False


# ---------------------------------------------------
# PLANNER CORE
# ---------------------------------------------------

class Planner:
    """
    Turns reasoning output into an executable multi-step plan.

    Input:
        reasoning = {
            "goal": Goal.ADD_TASK,
            "confidence": 0.8,
            "entities": {...}
        }

    Output:
        Plan(steps=[...])
    """

    def create_plan(self, reasoning: Dict[str, Any]) -> Plan:
        goal = reasoning.get("goal", Goal.UNKNOWN)
        entities = reasoning.get("entities", {})
        confidence = reasoning.get("confidence", 1.0)

        if goal == Goal.ADD_TASK:
            return self._plan_add_task(entities, confidence)

        if goal == Goal.COMPLETE_TASK:
            return self._plan_complete_task(entities, confidence)

        if goal == Goal.DELETE_TASK:
            return self._plan_delete_task(entities, confidence)

        if goal == Goal.VIEW_TASKS:
            return self._plan_view_tasks(confidence)

        return self._plan_fallback(confidence)

    # ---------------------------------------------------
    # ADD TASK
    # ---------------------------------------------------

    def _plan_add_task(self, entities: Dict[str, Any], confidence: float) -> Plan:
        return Plan(
            goal=Goal.ADD_TASK,
            confidence=confidence,
            steps=[
                PlanStep(
                    action="validate_input",
                    slots={
                        "title": entities.get("title"),
                        "due_date": entities.get("due_date"),
                        "category": entities.get("category", "personal")
                    },
                    description="Validate task input fields"
                ),
                PlanStep(
                    action="add_task",
                    slots={
                        "title": entities.get("title"),
                        "due_date": entities.get("due_date"),
                        "category": entities.get("category", "personal")
                    },
                    description="Create task in task system"
                ),
                PlanStep(
                    action="update_memory",
                    description="Sync memory and analytics"
                )
            ]
        )

    # ---------------------------------------------------
    # COMPLETE TASK
    # ---------------------------------------------------

    def _plan_complete_task(self, entities: Dict[str, Any], confidence: float) -> Plan:
        return Plan(
            goal=Goal.COMPLETE_TASK,
            confidence=confidence,
            steps=[
                PlanStep(
                    action="resolve_task",
                    slots={
                        "task_ref": entities.get("task_id") or entities.get("task_title")
                    },
                    description="Find best matching task"
                ),
                PlanStep(
                    action="complete_task",
                    slots={
                        "task_id": entities.get("task_id")
                    },
                    description="Mark task as completed"
                ),
                PlanStep(
                    action="update_memory",
                    description="Update learning signals"
                )
            ]
        )

    # ---------------------------------------------------
    # DELETE TASK
    # ---------------------------------------------------

    def _plan_delete_task(self, entities: Dict[str, Any], confidence: float) -> Plan:
        return Plan(
            goal=Goal.DELETE_TASK,
            confidence=confidence,
            requires_confirmation=True,
            steps=[
                PlanStep(
                    action="resolve_task",
                    slots={
                        "task_ref": entities.get("task_id") or entities.get("task_title")
                    },
                    description="Locate task to delete"
                ),
                PlanStep(
                    action="confirm_delete",
                    slots={
                        "task_id": entities.get("task_id")
                    },
                    description="Ask user confirmation before deletion"
                ),
                PlanStep(
                    action="delete_task",
                    slots={
                        "task_id": entities.get("task_id")
                    },
                    description="Delete task permanently"
                ),
                PlanStep(
                    action="update_memory",
                    description="Sync deletion to memory system"
                )
            ]
        )

    # ---------------------------------------------------
    # VIEW TASKS
    # ---------------------------------------------------

    def _plan_view_tasks(self, confidence: float) -> Plan:
        return Plan(
            goal=Goal.VIEW_TASKS,
            confidence=confidence,
            steps=[
                PlanStep(
                    action="list_tasks",
                    description="Fetch all user tasks"
                ),
                PlanStep(
                    action="format_tasks",
                    description="Convert tasks into natural language response"
                )
            ]
        )

    # ---------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------

    def _plan_fallback(self, confidence: float) -> Plan:
        return Plan(
            goal=Goal.UNKNOWN,
            confidence=confidence,
            steps=[
                PlanStep(
                    action="ask_user",
                    slots={
                        "prompt": "I’m not sure what you mean. Can you clarify?"
                    },
                    description="Fallback clarification"
                )
            ]
        )

    # ---------------------------------------------------
    # DEBUG / INSPECTION TOOL
    # ---------------------------------------------------

    def explain(self, plan: Plan) -> str:
        out = [
            f"Goal: {plan.goal}",
            f"Confidence: {plan.confidence}",
            f"Requires confirmation: {plan.requires_confirmation}",
            "",
            "Steps:"
        ]

        for i, step in enumerate(plan.steps, 1):
            out.append(f"{i}. {step.action} → {step.description}")

        return "\n".join(out)