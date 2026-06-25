"""
nlp/pipeline.py

NLUPipeline — single entry point for all NLU.

Replaces BOTH ReasoningEngine AND SlotExtractor with one object.

Usage
─────
    from nlp.pipeline import NLUPipeline

    nlu = NLUPipeline()               # load once at startup
    result = nlu.parse("add task to buy groceries by tomorrow")

    result.intent    # Goal.ADD_TASK
    result.title     # "Buy groceries"
    result.due_date  # "2025-XX-XX"
    result.category  # "personal" (or None if undetected)
    result.task_ref  # None

DialogueManager migration
─────────────────────────
Old code:
    analysis  = self.reasoning.analyze(text)   # ReasoningEngine
    intent    = analysis["goal"]
    re_entities = analysis.get("entities", {})
    extracted = self._extract_slots(text_clean) # SlotExtractor
    slots = {**extracted, **re_entities}        # messy merge

New code:
    result = self.nlu.parse(text)
    # result.intent, result.title, result.due_date, result.category, result.task_ref
    # All typed, all from one source.

Backward-compatible shim
─────────────────────────
NLUPipeline.analyze(text) returns the old dict shape so you can
swap out ReasoningEngine without touching DialogueManager on day one:

    engine = NLUPipeline()
    # works as a drop-in:
    analysis = engine.analyze(text)
    intent   = analysis["goal"]       # still works
"""
from __future__ import annotations

import spacy

# Register custom components before loading the pipeline
import nlp.components.intent    # noqa: F401  — registers "intent_classifier"
import nlp.components.slots     # noqa: F401  — registers "slot_filler"
import nlp.components.category  # noqa: F401  — registers "category_detector"

from nlp.parse_result import ParseResult
from brain.goal_types import Goal


class NLUPipeline:
    """
    Wraps a spaCy pipeline with three custom components:
      en_core_web_sm  →  intent_classifier  →  slot_filler  →  category_detector

    Instantiate once at startup (model load is expensive).
    .parse() is thread-safe for read-only use.
    """

    _MODEL = "en_core_web_sm"

    def __init__(self):
        self._nlp = spacy.load(self._MODEL)

        # Add our components after the base model's tagger/parser/ner
        self._nlp.add_pipe("intent_classifier", last=True)
        self._nlp.add_pipe("slot_filler",       last=True)
        self._nlp.add_pipe("category_detector", last=True)

    # ── Primary API ───────────────────────────────────────────────────────────

    def parse(self, text: str) -> ParseResult:
        """
        Parse raw user text and return a typed ParseResult.
        This is the method you should use in new code.
        """
        doc = self._nlp(text.strip())
        return ParseResult(
            intent     = doc._.intent,
            confidence = doc._.confidence,
            title      = doc._.title,
            due_date   = doc._.due_date,
            category   = doc._.category,
            task_ref   = doc._.task_ref,
            raw_text   = text,
        )

    # ── Backward-compatible shim ──────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        """
        Drop-in replacement for ReasoningEngine.analyze().
        Returns the same dict shape so DialogueManager works unchanged
        while you migrate incrementally.

        Old shape:
            {
              "goal": Goal.ADD_TASK,
              "confidence": 0.85,
              "entities": {
                  "title": "Buy groceries",
                  "due_date": "2025-08-20",
                  "task_ref": None,
              }
            }
        """
        result = self.parse(text)
        return {
            "goal":       result.intent,
            "confidence": result.confidence,
            "entities": {
                "title":      result.title,
                "due_date":   result.due_date,
                "category":   result.category,
                "task_ref":   result.task_ref,
                "raw_text":   result.raw_text,
            },
        }

    # ── Pipeline inspection ───────────────────────────────────────────────────

    def pipe_names(self) -> list[str]:  # pragma: no cover
        return self._nlp.pipe_names