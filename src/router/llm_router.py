from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class RouteResult:
    intent: str
    confidence: float
    entities: Dict[str, Any]
    requires_plan: bool


class LLMRouter:
    """
    Lightweight router (can later plug in LLM).
    """

    def classify(self, text: str) -> RouteResult:
        text_lower = text.lower()

        # --- minimal fallback logic (replace later with LLM) ---
        if "add" in text_lower or "remind" in text_lower:
            return RouteResult(
                intent="add_task",
                confidence=0.7,
                entities={"raw": text},
                requires_plan=True
            )

        if "delete" in text_lower or "remove" in text_lower:
            return RouteResult(
                intent="delete_task",
                confidence=0.7,
                entities={"raw": text},
                requires_plan=True
            )

        if "show" in text_lower or "list" in text_lower:
            return RouteResult(
                intent="view_tasks",
                confidence=0.8,
                entities={},
                requires_plan=False
            )

        return RouteResult(
            intent="chat",
            confidence=0.5,
            entities={"raw": text},
            requires_plan=False
        )