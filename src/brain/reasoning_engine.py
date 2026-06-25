import re
from brain.goal_types import Goal


class ReasoningEngine:
    """
    Rule-based intent classifier.
    Converts raw text → {goal, confidence, entities}.

    Rules are ordered from most-specific to least-specific so that
    shorter trigger words (e.g. "done") don't shadow longer phrases.
    """

    # ── Pre-compiled patterns ─────────────────────────────────────────────────

    # "mark X as done / complete" where X is a task name
    _MARK_AS_DONE = re.compile(
        r"mark\s+(.+?)\s+as\s+(done|complete|completed|finished)",
        re.IGNORECASE,
    )

    # "i've completed / finished / done X" — capture the trailing name.
    # Also catches: "i also finished X", "i finish X", "i just completed X"
    _COMPLETED_NAMED = re.compile(
        r"(?:i'?ve?|i\s+have|i)\s+(?:also\s+|just\s+|already\s+)?"
        r"(?:completed?|finished?|done|finish)\s+(.+)",
        re.IGNORECASE,
    )

    # Trailing filler to strip from extracted task names
    # e.g. "go to campus task already" → "go to campus"
    _TASK_NAME_TAIL = re.compile(
        r"\s+(task|already|as well|too|just now|now|today|yesterday)[\s,]*$",
        re.IGNORECASE,
    )

    # Vague add-intent sentences that carry no title payload.
    # "i want to add a task", "i'd like to create a task", "can you add a task" …
    _VAGUE_ADD = re.compile(
        r"^(i('d| would)? (want|like|need) to |can (i|you) |please )?"
        r"(add|create|make|set up) (a |an |new )?(task|reminder|to.?do)s?$",
        re.IGNORECASE,
    )

    # Vague view-intent sentences.
    # "can i see my tasks?", "show me my tasks", "let me see my tasks" …
    _VAGUE_VIEW = re.compile(
        r"(can i|let me|show me|i want to|i('d| would) like to) "
        r"(see|view|check|look at) (my )?(tasks?|to.?dos?|list)",
        re.IGNORECASE,
    )

    def analyze(self, text: str) -> dict:
        t = text.lower().strip()

        # ── GREETING ──────────────────────────────────────────────────────────
        if t in {"hi", "hello", "hey", "good morning", "good afternoon",
                 "good evening", "howdy", "sup", "what's up", "yo"}:
            return {"goal": Goal.GREETING, "confidence": 0.99, "entities": {}}

        # ── ACKNOWLEDGEMENT ───────────────────────────────────────────────────
        if t in {"ok", "okay", "alright", "sure", "thanks", "thank you",
                 "got it", "cool", "nice", "great", "sounds good", "perfect"}:
            return {"goal": Goal.ACKNOWLEDGEMENT, "confidence": 0.99, "entities": {}}

        # ── GET COMPLETED TASKS ───────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "completed tasks", "finished tasks", "tasks completed",
            "done tasks", "tasks i've done", "tasks i have done",
        ]):
            return {"goal": Goal.GET_COMPLETED_TASKS, "confidence": 0.95, "entities": {}}

        # ── COMPLETE TASK — "mark X as done" (most specific first) ───────────
        mark_match = self._MARK_AS_DONE.search(text)
        if mark_match:
            return {
                "goal": Goal.COMPLETE_TASK,
                "confidence": 0.9,
                "entities": {
                    "raw_text": text,
                    "task_name": mark_match.group(1).strip(),
                    "task_ref": self._extract_reference(text),
                },
            }

        # ── COMPLETE TASK — "i've completed X" / "i also finished X" ───────────
        done_match = self._COMPLETED_NAMED.search(text)
        if done_match:
            raw_name = done_match.group(1).strip()
            task_name = self._TASK_NAME_TAIL.sub("", raw_name).strip()
            return {
                "goal": Goal.COMPLETE_TASK,
                "confidence": 0.85,
                "entities": {
                    "raw_text": text,
                    "task_name": task_name,
                    "task_ref": self._extract_reference(text),
                },
            }

        # ── COMPLETE TASK — bare trigger words ────────────────────────────────
        if any(word in t for word in [
            "finished", "completed", "i did", "i've done",
            "i completed", "mark as done", "mark done", "finish",
        ]):
            return {
                "goal": Goal.COMPLETE_TASK,
                "confidence": 0.75,
                "entities": {
                    "raw_text": text,
                    "task_ref": self._extract_reference(text),
                },
            }

        # ── DELETE TASK ───────────────────────────────────────────────────────
        if any(word in t for word in [
            "delete", "remove", "cancel", "take it out", "erase", "drop task",
        ]):
            return {
                "goal": Goal.DELETE_TASK,
                "confidence": 0.8,
                "entities": {
                    "raw_text": text,
                    "task_ref": self._extract_reference(text),
                },
            }

        # ── VIEW TASKS — vague phrasing ("can i see my tasks?") ──────────────
        if self._VAGUE_VIEW.search(t):
            return {"goal": Goal.LIST_TASKS, "confidence": 0.9, "entities": {}}

        # ── VIEW TASKS — explicit keywords ────────────────────────────────────
        if any(phrase in t for phrase in [
            "show task", "show my task", "list task", "my tasks",
            "what are my tasks", "tasks today", "show all tasks", "full list",
            "show tasks", "show my tasks", "show task list", "list tasks",
            "display tasks", "view tasks", "see my tasks", "all my tasks",
        ]):
            return {"goal": Goal.LIST_TASKS, "confidence": 0.9, "entities": {}}

        # ── ADD TASK — vague intent, no title ("i want to add a task") ─────────
        if self._VAGUE_ADD.match(t):
            return {
                "goal": Goal.ADD_TASK,
                "confidence": 0.85,
                "entities": {"title": None, "due_date": None},
            }

        # ── ADD TASK — has a title payload ────────────────────────────────────
        # Guard: "i need to" / "i have to" in a question = NOT a task creation
        _is_question = t.endswith("?") or t.startswith(("what", "when", "where",
                                                          "who", "why", "how",
                                                          "which", "is ", "are ",
                                                          "do ", "does ", "did ",
                                                          "can ", "could ", "should "))
        if any(word in t for word in [
            "remind me", "add task", "create task", "new task",
            "remember to", "schedule", "add a task", "add reminder",
        ]) or (not _is_question and any(w in t for w in ["i need to", "i have to"])):
            return {
                "goal": Goal.ADD_TASK,
                "confidence": 0.85,
                "entities": {
                    "title": self._extract_task_title(text),
                    "due_date": self._extract_date(text),
                },
            }

        # ── PRODUCTIVITY ──────────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "productivity score", "my productivity", "how productive am i",
        ]):
            return {"goal": Goal.PRODUCTIVITY_SCORE, "confidence": 0.9, "entities": {}}

        # ── OVERDUE ───────────────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "overdue", "late tasks", "missed tasks", "past due",
        ]):
            return {"goal": Goal.GET_OVERDUE_TASKS, "confidence": 0.9, "entities": {}}

        # ── RECOMMENDATIONS ───────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "recommend task", "what should i do next", "next task",
            "what should i do", "what to do today", "what to do next",
            "what's next", "what is next", "suggest a task", "suggest tasks",
            "what should i work on", "what can i do today",
            "which one should i", "what should i tackle", "what to tackle",
            "priorit", "which task first", "what's my next task",
            "what is my next task", "my next task",
            "what do i need to do", "what do i have to do",
            "what needs to be done", "what's on my list",
        ]):
            return {"goal": Goal.GET_RECOMMENDED_TASKS, "confidence": 0.9, "entities": {}}
        
        # Check tasks
        if any(w in t for w in [
            ""
        ]): return {"goal": Goal.CHECK_TASK, "confidence": 0.85, "entities": {}}

        # ── HABITS ────────────────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "analyze habits", "my habits", "habit report",
        ]):
            return {"goal": Goal.ANALYZE_HABITS, "confidence": 0.9, "entities": {}}

        # ── USER NAME ─────────────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "what's my name", "what is my name", "who am i",
            "do you know my name",
        ]):
            return {"goal": Goal.GET_USER_NAME, "confidence": 0.95, "entities": {}}

        # ── CHAT FALLBACK ─────────────────────────────────────────────────────
        return {"goal": Goal.CHAT, "confidence": 0.4, "entities": {"raw_text": text}}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_reference(self, text: str):
        match = re.search(r"\b(\d+)\b", text)
        return match.group(1) if match else None

    def _extract_task_title(self, text: str) -> str | None:
        cleaned = re.sub(
            r"(remind me to|remind me|add task( to)?|create task( to)?|new task"
            r"|i need to|i have to|remember to|schedule|add a task( to)?|"
            r"add a reminder( to)?|add reminder( to)?)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        # strip bare leading "to" left behind
        cleaned = re.sub(r"^to\s+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(
            r"\s*(by|on|before|due)\s+(today|tomorrow|next week|\d{4}-\d{2}-\d{2})$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        return cleaned.capitalize() if len(cleaned) > 1 else None

    def _extract_date(self, text: str) -> str | None:
        t = text.lower()
        for k, v in {"today": "today", "tomorrow": "tomorrow", "next week": "next week"}.items():
            if k in t:
                return v
        return None