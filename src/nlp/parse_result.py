"""
nlp/parse_result.py

Typed output from NLUPipeline.parse().
Replaces the raw dict that ReasoningEngine.analyze() returned.

DialogueManager reads these fields directly — no more .get("entities", {})
and no more merging two dicts from two different sources.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from brain.goal_types import Goal


@dataclass
class ParseResult:
    # Core intent — always set, never None
    intent: Goal            = Goal.UNKNOWN

    # Slots — None means "not found in this utterance"
    title: str | None       = None
    category: str | None    = None
    due_date: str | None    = None   # ISO string e.g. "2025-08-20", or "tomorrow"
    task_ref: str | None    = None   # bare numeric ID from user input

    # How confident the intent classifier is (0.0–1.0)
    confidence: float       = 0.0

    # Original text, preserved for downstream fuzzy matching / logging
    raw_text: str           = ""

    def __repr__(self) -> str:  # pragma: no cover
        slots = {
            k: v for k, v in [
                ("title",    self.title),
                ("category", self.category),
                ("due_date", self.due_date),
                ("task_ref", self.task_ref),
            ] if v is not None
        }
        return (
            f"ParseResult(intent={self.intent.name}, "
            f"confidence={self.confidence:.2f}, "
            f"slots={slots})"
        )