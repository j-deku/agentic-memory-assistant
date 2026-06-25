"""
nlp/components/intent.py

IntentClassifier — spaCy pipeline component.

Replaces ReasoningEngine's keyword/regex cascade with dependency-tree
and POS-tag aware classification.  Sets doc._.intent and doc._.confidence.

How it works
────────────
Rather than matching surface strings ("does the text contain 'add'?"),
we look at grammatical structure:

  ADD_TASK:       root verb is an imperative/directive AND there's an
                  object noun-phrase that isn't a task-management verb
                  e.g. "add a task to buy groceries"
                       "remind me to call the dentist"
                       "i need to submit the report"

  COMPLETE_TASK:  root verb signals completion (finish, complete, done,
                  mark) OR auxiliary "have" + past-participle pattern
                  e.g. "I finished the report"
                       "mark task 3 as done"

  DELETE_TASK:    root verb is delete/remove/cancel/erase

  LIST_TASKS:     root verb is show/list/display/see/view + task object,
                  or bare question "what are my tasks"

  GREETING / ACK: single-token or known fixed phrases — checked first
                  before any tree traversal (cheap early exit)

Fallback chain:  DEP-tree → POS heuristics → keyword set → CHAT/UNKNOWN
"""
from __future__ import annotations

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from brain.goal_types import Goal


# ── Custom attribute registration ─────────────────────────────────────────────

if not Doc.has_extension("intent"):
    Doc.set_extension("intent", default=Goal.UNKNOWN)

if not Doc.has_extension("confidence"):
    Doc.set_extension("confidence", default=0.0)


# ── Fixed-phrase lookup tables (cheap O(1) checks done before tree walk) ──────

_GREETINGS = frozenset({
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "sup", "what's up", "yo",
})

_ACKS = frozenset({
    "ok", "okay", "alright", "sure", "thanks", "thank you",
    "got it", "cool", "nice", "great", "sounds good", "perfect",
    "noted", "understood",
})

# Root lemmas that signal task completion
_COMPLETE_VERBS = frozenset({
    "finish", "complete", "done", "accomplish", "wrap",
    "finalize", "submit", "close",
})

# Root lemmas that signal task deletion
_DELETE_VERBS = frozenset({
    "delete", "remove", "cancel", "erase", "drop", "clear", "eliminate",
})

# Root lemmas that signal listing / viewing
_VIEW_VERBS = frozenset({
    "show", "list", "display", "see", "view", "get", "give",
    "tell", "fetch", "check",
})

# Root lemmas that signal adding / creating
_ADD_VERBS = frozenset({
    "add", "create", "make", "set", "schedule", "remind",
    "note", "record", "put", "track",
})

# "i need to X", "i have to X", "i want to X" — the X's lemma decides intent.
# These surface forms are NOT add-intents on their own; we need the xcomp.
_MODAL_HEADS = frozenset({"need", "have", "want", "like", "would", "plan", "intend"})

# Phrases that signal the user is talking about the task *list*, not a task name.
# Used to avoid "what are my tasks" → ADD_TASK.
_LIST_OBJECTS = frozenset({
    "task", "tasks", "todo", "todos", "to-do", "to-dos",
    "list", "reminder", "reminders",
})

# Productivity / analytics intents keyed on salient lemmas
_PRODUCTIVITY_PHRASES = frozenset({
    "productivity", "productive", "score",
})
_OVERDUE_PHRASES = frozenset({
    "overdue", "late", "missed", "past due",
})
_RECOMMEND_PHRASES = frozenset({
    "recommend", "suggest", "next", "priorit",
    "what should", "what to do",
})
_HABIT_PHRASES = frozenset({
    "habit", "habits", "analyse habits", "analyze habits",
})


# ── Helper ────────────────────────────────────────────────────────────────────

def _lemmas(doc: Doc) -> set[str]:
    """Return all token lemmas (lower-cased) in the doc."""
    return {t.lemma_.lower() for t in doc}


def _text_lower(doc: Doc) -> str:
    return doc.text.lower().strip()


def _has_task_object(doc: Doc) -> bool:
    """
    Return True if any token in the doc is a task-list noun
    (tasks / todo / list / reminder …).  Used to distinguish
    "show my tasks" (LIST) from "show me how to cook" (CHAT).
    """
    return bool(_lemmas(doc) & _LIST_OBJECTS)


def _root_verb(doc: Doc):
    """Return the syntactic root token if it's a VERB, else None."""
    for token in doc:
        if token.dep_ == "ROOT":
            return token if token.pos_ in ("VERB", "AUX") else None
    return None


def _xcomp_verb(doc: Doc):
    """
    For "I need to buy X", "I want to delete Y" etc., return the
    xcomp (open clausal complement) verb token, which carries the
    *real* intent.
    """
    for token in doc:
        if token.dep_ == "xcomp" and token.pos_ == "VERB":
            return token
    return None


# ── Component factory ─────────────────────────────────────────────────────────

@Language.factory("intent_classifier")
def create_intent_classifier(nlp: Language, name: str):
    return IntentClassifier(nlp, name)


class IntentClassifier:
    """
    spaCy v3 pipeline component.
    Sets doc._.intent (Goal) and doc._.confidence (float).
    """

    def __init__(self, nlp: Language, name: str):
        self.name = name

    # ── Main entry ────────────────────────────────────────────────────────────

    def __call__(self, doc: Doc) -> Doc:
        intent, conf = self._classify(doc)
        doc._.intent     = intent
        doc._.confidence = conf
        return doc

    # ── Classification logic ──────────────────────────────────────────────────

    def _classify(self, doc: Doc) -> tuple[Goal, float]:
        t = _text_lower(doc)
        lemmas = _lemmas(doc)

        # ── 1. Fixed phrases (O(1), no tree needed) ───────────────────────────

        if t in _GREETINGS:
            return Goal.GREETING, 0.99

        if t in _ACKS:
            return Goal.ACKNOWLEDGEMENT, 0.99

        # ── 2. Analytics intents (keyword presence is sufficient here) ────────
        #    These are rarely ambiguous with task CRUD, so keyword check is fine.

        if any(p in t for p in _OVERDUE_PHRASES):
            return Goal.GET_OVERDUE_TASKS, 0.90

        if any(p in t for p in _PRODUCTIVITY_PHRASES):
            return Goal.PRODUCTIVITY_SCORE, 0.90

        if any(p in t for p in _HABIT_PHRASES):
            return Goal.ANALYZE_HABITS, 0.90

        if any(p in t for p in _RECOMMEND_PHRASES):
            return Goal.GET_RECOMMENDED_TASKS, 0.85

        if any(p in t for p in [
            "completed tasks", "finished tasks", "tasks i've done",
            "tasks i have done", "done tasks",
        ]):
            return Goal.GET_COMPLETED_TASKS, 0.95

        if any(p in t for p in [
            "what's my name", "what is my name", "who am i",
            "do you know my name",
        ]):
            return Goal.GET_USER_NAME, 0.95

        # ── 3. DEP-tree classification for CRUD intents ───────────────────────

        root = _root_verb(doc)

        # "mark X as done" — special construction, check before generic complete
        if "mark" in lemmas and ("done" in t or "complete" in t or "finish" in t):
            return Goal.COMPLETE_TASK, 0.92

        # Resolve the *effective* verb: for "I need to delete X", it's "delete"
        # KEY RULE: when root is a modal (need/have/want), the xcomp is ALWAYS
        # the effective verb — even if that xcomp is a completion verb like
        # "finish" or "submit". "I need to finish X" = ADD_TASK (future intent),
        # NOT COMPLETE_TASK (past action). Context is the subject "I need to".
        effective_verb = None
        root_is_modal = False
        if root is not None:
            if root.lemma_.lower() in _MODAL_HEADS:
                root_is_modal = True
                xc = _xcomp_verb(doc)
                effective_verb = xc if xc is not None else root
            else:
                effective_verb = root

        if effective_verb is not None:
            ev_lemma = effective_verb.lemma_.lower()

            # When root is modal, the xcomp expresses future intent → ADD_TASK
            # regardless of whether the xcomp lemma is a "completion" word.
            # "I need to finish X" ≠ "I finished X"
            if root_is_modal:
                if ev_lemma in _DELETE_VERBS:
                    return Goal.DELETE_TASK, 0.88
                if ev_lemma in _VIEW_VERBS and _has_task_object(doc):
                    return Goal.LIST_TASKS, 0.85
                # Everything else under a modal = something the user needs to do
                return Goal.ADD_TASK, 0.80

            # Non-modal root — verb lemma decides directly
            if ev_lemma in _DELETE_VERBS:
                return Goal.DELETE_TASK, 0.88

            if ev_lemma in _COMPLETE_VERBS:
                return Goal.COMPLETE_TASK, 0.88

            if ev_lemma in _VIEW_VERBS and _has_task_object(doc):
                return Goal.LIST_TASKS, 0.88

            if ev_lemma in _ADD_VERBS:
                return Goal.ADD_TASK, 0.85

        # ── 3b. Keyword safety net — handles cases where en_core_web_sm
        #        misparses the root (e.g. "delete task 3" → root tagged NOUN)
        if not root_is_modal:
            if any(tok.lemma_.lower() in _DELETE_VERBS for tok in doc if tok.pos_ in ("VERB", "NOUN", "ADJ")):
                # Verify a delete-word actually appears as a token lemma
                delete_lemmas = {tok.lemma_.lower() for tok in doc}
                if delete_lemmas & _DELETE_VERBS:
                    return Goal.DELETE_TASK, 0.80

            complete_past_tags = {"VBD", "VBN"}  # past tense / past participle
            for tok in doc:
                if tok.lemma_.lower() in _COMPLETE_VERBS and tok.tag_ in complete_past_tags:
                    return Goal.COMPLETE_TASK, 0.80

        # ── 4. POS heuristic fallback ─────────────────────────────────────────
        #    If we still don't know, look at the overall POS composition.

        # "show tasks", "list tasks" without a detected root verb
        if _has_task_object(doc) and any(
            t.startswith(v) for v in ("show", "list", "display", "view", "see")
        ):
            return Goal.LIST_TASKS, 0.75

        # Question about tasks → LIST
        if len(doc) > 0 and doc[0].tag_ in ("WP", "WRB", "WDT") and _has_task_object(doc):
            return Goal.LIST_TASKS, 0.70

        # ── 5. True fallback ──────────────────────────────────────────────────

        return Goal.CHAT, 0.35