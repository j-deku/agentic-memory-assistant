from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Step:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    intent: str
    steps: List[Step]
    slots: Dict[str, Any]
    requires_user_input: bool = False
    next_slot: Optional[str] = None


class Planner:

    def generate(self, route, state=None) -> Plan:
        intent = route.intent
        entities = route.entities

        # ---------------- ADD TASK ----------------
        if intent == "add_task":

            if not entities.get("title"):
                return Plan(
                    intent="add_task",
                    steps=[Step("ask_title")],
                    slots=entities,
                    requires_user_input=True,
                    next_slot="title"
                )

            if not entities.get("due_date"):
                return Plan(
                    intent="add_task",
                    steps=[Step("ask_due_date")],
                    slots=entities,
                    requires_user_input=True,
                    next_slot="due_date"
                )

            return Plan(
                intent="add_task",
                steps=[Step("execute_add_task", entities)],
                slots=entities
            )

        # ---------------- DELETE TASK ----------------
        if intent == "delete_task":

            if not entities.get("task_id"):
                return Plan(
                    intent="delete_task",
                    steps=[Step("resolve_task")],
                    slots=entities,
                    requires_user_input=True,
                    next_slot="task_id"
                )

            return Plan(
                intent="delete_task",
                steps=[Step("execute_delete_task", entities)],
                slots=entities
            )

        # ---------------- VIEW ----------------
        if intent == "view_tasks":
            return Plan(
                intent="view_tasks",
                steps=[Step("execute_list_tasks")],
                slots={}
            )

        # ---------------- FALLBACK ----------------
        return Plan(
            intent="chat",
            steps=[Step("llm_response")],
            slots={}
        )