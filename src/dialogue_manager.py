"""
dialogue_manager.py

Conversational orchestration layer.

Changes from previous version:
  - Replaced ReasoningEngine + SlotExtractor with NLUPipeline (single source of truth)
  - Fixed _best_match: fuzzy fallback was inside the outer loop, causing it to
    return on the first task regardless of score; all match passes now complete
    before any return
  - Removed the two-dict merge (extracted + re_entities) — ParseResult fields
    are used directly
  - Removed stale regex noise patterns (_COMPLETE_NOISE, _DELETE_NOISE) that
    were only needed because SlotExtractor couldn't parse properly
  - _add_recv_title now uses NLUPipeline instead of SlotExtractor
  - _add_recv_due now uses NLUPipeline instead of SlotExtractor
"""

from datetime import datetime
import random
import re

from brain.goal_types import Goal
from core.state import DialogueState
from services.task_service import TaskService
from nlp.pipeline import NLUPipeline
from nlp.response_engine import ResponseEngine
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Titles that are clearly mis-saved intent phrases — filter from recommendations
_JUNK_TITLE = re.compile(
    r"^(add (a |an )?(task|reminder|to.?do)"
    r"|create (a |an )?(task|reminder)"
    r"|i (want|would like|need|have) to"
    r"|remind(er)?"
    r"|new task"
    r"|untitled"
    r"|can (i|you)"
    r")(\s|$)",
    re.IGNORECASE,
)

_KNOWN_CATEGORIES = ("work", "personal", "health", "school", "study", "learning")

# Minimum fuzzy score to accept a match (0-100)
_FUZZY_THRESHOLD = 75


def _is_empty_title(title: str | None) -> bool:
    """Return True if the NLU pipeline returned no usable task title."""
    if not title:
        return True
    return len(title.strip()) < 2


def _task_title(task) -> str:
    return (task[1] if isinstance(task, (list, tuple)) else task.get("title", "")).lower()


def _task_id(task):
    return task[0] if isinstance(task, (list, tuple)) else task.get("id")


# ---------------------------------------------------------------------------
# DialogueManager
# ---------------------------------------------------------------------------

class DialogueManager:
    """
    Conversational orchestration layer.

    Add-task flow (multi-turn):
        1. Title    -> "What's the task?"
        2. Category -> "What category? (work / personal / health / other)"
        3. Due date -> "When is it due? (or 'skip')"
        -> save

    Complete / delete use fuzzy name matching before falling back to listing.
    """

    def __init__(self, task_fns: dict, reasoning=None, user_name: str = ""):
        self.state     = DialogueState()
        self.tasks     = TaskService(task_fns)
        self.user_name = user_name
        self.responses = ResponseEngine()

        # Accept either a pre-built NLUPipeline or a legacy ReasoningEngine.
        if isinstance(reasoning, NLUPipeline):
            self.nlu = reasoning
        else:
            self.nlu = NLUPipeline()

        self.ai = getattr(reasoning, "ai", None)

    # Public entry point

    def process(self, text: str) -> str:
        self.state.add_turn("user", text)

        text_clean = text.strip()

        result = self.nlu.parse(text_clean)

        # User changed topic while answering a slot?
        if self.state.awaiting_slot:

            if (
                self.state.intent
                and result.intent
                and result.intent != self.state.intent
            ):
                self.state.reset()

        response = self._route(
            result.intent,
            text_clean,
            result
        )

        self.state.add_turn("assistant", response)

        return response

    # Routing

    def _route(self, intent, text, result):
        if self.state.awaiting_slot == "add_task_title":
            return self._add_recv_title(text)

        if self.state.awaiting_slot == "add_task_category":
            return self._add_recv_category(text)

        if self.state.awaiting_slot == "add_task_due":
            return self._add_recv_due(text)

        if self.state.awaiting_slot == "complete_task":
            return self._complete_from_followup(text)

        if self.state.awaiting_slot == "delete_task":
            return self._delete_from_followup(text)

        if intent == Goal.ADD_TASK:
            return self._handle_add(result, text)

        if intent == Goal.LIST_TASKS:
            return self._handle_view()

        if intent == Goal.COMPLETE_TASK:
            return self._handle_complete(result, text)

        if intent == Goal.DELETE_TASK:
            return self._handle_delete(result, text)

        if intent == Goal.GET_OVERDUE_TASKS:
            return self._handle_overdue()

        if intent == Goal.PRODUCTIVITY_SCORE:
            return self._handle_productivity()

        if intent == Goal.GET_RECOMMENDED_TASKS:
            return self._handle_recommendations()

        if intent == Goal.ANALYZE_HABITS:
            return self._handle_habits()

        if intent == Goal.GET_USER_NAME:
            return f"Your name is {self.user_name}."

        if intent == Goal.GREETING:
            return random.choice([
                f"Hey {self.user_name}! What would you like to do?",
                f"Hi {self.user_name}! How can I help?",
                "Hello! Ready when you are.",
            ])

        if intent == Goal.ACKNOWLEDGEMENT:
            return random.choice([
                "Alright.",
                f"No problem, {self.user_name}. Is there something you want me to check for you?",
                f"Sure, {self.user_name}. What else?",
            ])

        return self._fallback(text)

    # ADD TASK - multi-turn flow

    def _handle_add(self, result, raw_text: str) -> str:
        title = result.title if result else None

        if not result.title:
            self.state.intent = Goal.ADD_TASK
            self.state.awaiting_slot = "add_task_title"

            self.state.pending_slots["due_date"] = result.due_date

            return "What should I remind you about?"

        if _is_empty_title(title):
            self.state.awaiting_slot = "add_task_title"
            self.state.intent = Goal.ADD_TASK
            self.state.pending_slots = {}
            return "Sure! What would you like to call this task?"

        self.state.pending_slots = {
            "title":    title,
            "category": result.category if result else None,
            "due_date": result.due_date if result else None,
        }
        return self._next_add_step()

    def _add_recv_title(self, text: str) -> str:
        result = self.nlu.parse(text)
        title = result.title or text.strip().capitalize()

        if not title or len(title) < 2:
            return "Please give the task a name."

        self.state.pending_slots["title"] = title.capitalize()
        self.state.awaiting_slot = "add_task_category"
        return (
            "Got it! What category is this task?\n"
            "  work / personal / health / school / other  (or press Enter to skip)"
        )

    def _add_recv_category(self, text: str) -> str:
        t = text.strip().lower()
        if t in ("skip", "", "other"):
            category = "personal"
        elif t in _KNOWN_CATEGORIES:
            category = t
        else:
            category = next((c for c in _KNOWN_CATEGORIES if c in t), "personal")
        self.state.pending_slots["category"] = category
        self.state.awaiting_slot = "add_task_due"
        return "When is it due? (e.g. 'today', 'tomorrow', 'next week', '2025-08-20', or 'skip')"

    def _add_recv_due(self, text: str) -> str:
        t = text.strip().lower()
        if t in ("skip", "none", "no", ""):
            due = None
        else:
            result = self.nlu.parse(text)
            due = result.due_date or text.strip()
        self.state.pending_slots["due_date"] = due
        return self._finalize_add()

    def _next_add_step(self) -> str:
        p = self.state.pending_slots
        if not p.get("category"):
            self.state.awaiting_slot = "add_task_category"
            return (
                "What category is this task?\n"
                "  work / personal / health / school / other  (or 'skip')"
            )
        if p.get("due_date") is None:
            self.state.awaiting_slot = "add_task_due"
            return "When is it due? (e.g. 'tomorrow', 'next week', or 'skip')"
        return self._finalize_add()

    def _finalize_add(self) -> str:
        p = self.state.pending_slots
        title    = p.get("title", "Untitled task")
        category = p.get("category") or "personal"
        due      = p.get("due_date")
        self.state.reset()
        self.tasks.add(title, category, due)
        self.state.last_action = f"added:{title}"
        self.state.last_action_time = datetime.now()
        return self.responses.task_added(title, due, category)

    # VIEW

    def _handle_view(self) -> str:
        tasks = self.tasks.list_tasks()
        self.state.reset()
        return self.responses.task_list(tasks)

    # COMPLETE

    def _handle_complete(self, result, raw_text: str) -> str:
        if result and result.task_ref:
            return self._do_complete_by_id(result.task_ref)

        referenced = self._resolve_pronoun_task(raw_text)
        if referenced:
            match = self._find_and_complete_by_name(referenced)
            if match:
                return match

        if result and result.title:
            match = self._find_and_complete_by_name(result.title)
            if match:
                return match

        match = self._find_and_complete_by_name(raw_text)
        if match:
            return match

        tasks = self.tasks.list_tasks()
        self.state.awaiting_slot = "complete_task"
        self.state.intent = Goal.COMPLETE_TASK
        return self._ask_for_task(tasks, action="complete")

    def _complete_from_followup(self, text: str) -> str:
        result = self._find_and_complete_by_name(text)
        if result:
            return result
        return "I couldn't find that task. Type part of the name or its number."

    def _find_and_complete_by_name(self, text: str):
        tasks = self.tasks.list_tasks()
        match = self._best_match(text, tasks)
        if match:
            tid, title = match
            self.tasks.complete(tid)
            self.state.last_task = title
            self.state.last_task_id = tid
            self.state.reset()
            return self.responses.task_completed(title)
        return None

    def _do_complete_by_id(self, task_id) -> str:
        self.tasks.complete(task_id)
        self.state.last_action = f"completed:{task_id}"
        self.state.last_action_time = datetime.now()
        self.state.reset()
        return self.responses.task_completed(task_id)

    # DELETE

    def _handle_delete(self, result, raw_text: str) -> str:
        if result and result.task_ref:
            return self._do_delete_by_id(result.task_ref)

        referenced = self._resolve_pronoun_task(raw_text)
        if referenced:
            match = self._find_and_delete_by_name(referenced)
            if match:
                return match

        if result and result.title:
            match = self._find_and_delete_by_name(result.title)
            if match:
                return match

        match = self._find_and_delete_by_name(raw_text)
        if match:
            return match

        tasks = self.tasks.list_tasks()
        self.state.awaiting_slot = "delete_task"
        self.state.intent = Goal.DELETE_TASK
        return self._ask_for_task(tasks, action="delete")

    def _delete_from_followup(self, text: str) -> str:
        result = self._find_and_delete_by_name(text)
        if result:
            return result
        return "I couldn't find that task. Type part of the name or its number."

    def _find_and_delete_by_name(self, text: str):
        tasks = self.tasks.list_tasks()
        match = self._best_match(text, tasks)
        if match:
            tid, title = match
            self.tasks.delete(tid)
            self.state.last_task = title
            self.state.last_task_id = tid
            self.state.reset()
            return self.responses.task_deleted(title)
        return None

    def _do_delete_by_id(self, task_id) -> str:
        self.tasks.delete(task_id)
        self.state.last_action = f"deleted:{task_id}"
        self.state.last_action_time = datetime.now()
        self.state.reset()
        return self.responses.task_deleted(task_id)

    # Other handlers

    def _handle_productivity(self) -> str:
        score = self.tasks.productivity_score()
        return f"Your productivity score is {score}."

    def _handle_overdue(self) -> str:
        tasks = self.tasks.overdue_tasks()
        return self.responses.overdue_tasks(tasks)

    def _handle_recommendations(self) -> str:
        tasks = self.tasks.recommended_tasks()
        clean_tasks = [
            t for t in tasks
            if not _JUNK_TITLE.match(_task_title(t).strip())
        ]
        return self.responses.recommended_tasks(clean_tasks)

    def _handle_habits(self) -> str:
        return str(self.tasks.analyze_habits())

    # Helpers

    def _resolve_pronoun_task(self, text: str) -> str | None:
        pronouns = {"it", "that", "that task", "the task", "this", "this task"}
        if text.lower().strip() in pronouns:
            return self.state.last_task
        return None

    def _best_match(self, query: str, tasks: list):
        """
        Return (tid, title) for the best matching task, or None.

        Priority order - ALL candidates collected before any return:
          1. Numeric ID  - exact digit match, returns immediately
          2. Exact title - returns immediately
          3. Starts-with - query is a prefix of title
          4. Substring   - query in title, or title in query
          5. Fuzzy       - rapidfuzz across ALL tasks, threshold 75

        Bug fixed: previously passes 3-5 were indented inside the outer
        loop, so we returned on the FIRST task that passed any check
        instead of finding the best match across all tasks. The fuzzy
        loop was also nested inside the first loop, running a full O(n)
        scan per iteration for an O(n^2) total with wrong semantics.
        """
        if not tasks or not query:
            return None

        q = query.lower().strip()

        # Pass 1: numeric ID
        if q.isdigit():
            for task in tasks:
                if str(_task_id(task)) == q:
                    return (_task_id(task), _task_title(task))

        # Pass 2, 3, 4: exact / starts-with / substring
        # Single loop, all three collected; exact short-circuits immediately.
        starts_with_match = None
        best_sub_match    = None
        best_sub_len      = 0

        for task in tasks:
            tid   = _task_id(task)
            title = _task_title(task)

            if title == q:
                return (tid, title)

            if title.startswith(q) and starts_with_match is None:
                starts_with_match = (tid, title)

            if (q in title or title in q) and len(title) > best_sub_len:
                best_sub_match = (tid, title)
                best_sub_len   = len(title)

        if starts_with_match:
            return starts_with_match

        if best_sub_match:
            return best_sub_match

        # Pass 5: fuzzy - single loop over all tasks, keep global best
        best_score = 0
        best_fuzzy = None

        for task in tasks:
            tid   = _task_id(task)
            title = _task_title(task)
            score = fuzz.ratio(q, title)
            if score > best_score:
                best_score = score
                best_fuzzy = (tid, title)

        if best_score >= _FUZZY_THRESHOLD:
            return best_fuzzy

        return None

    def _ask_for_task(self, tasks, action="complete") -> str:
        if not tasks:
            return "You have no tasks right now."
        lines = [f"Which task do you want to {action}?"]
        for t in tasks[:8]:
            lines.append(f"  {_task_id(t)}. {_task_title(t).capitalize()}")
        return "\n".join(lines)

    def _fallback(self, text: str) -> str:
        lower = text.lower()

        if any(p in lower for p in [
            "what should i do", "what to do today", "what's next",
            "what next", "suggest", "recommend",
        ]):
            return self._handle_recommendations()

        if self.ai:
            return self.ai.chat(text, history=self.state.history)

        if "what can you do" in lower or "help" in lower:
            return (
                "I can help you:\n"
                "• Add tasks (with category and due date)\n"
                "• Complete or delete tasks by name\n"
                "• Show pending, overdue, or recommended tasks\n"
                "• Check your productivity score"
            )

        return (
            "I didn't quite catch that. "
            "Try 'add task', 'show tasks', or 'what should I do next'."
        )