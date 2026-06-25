import re
from datetime import datetime, timedelta


class SlotExtractor:

    # ── Noise patterns ────────────────────────────────────────────────────────

    # Phrases to strip before treating what remains as a task title on an ADD
    _ADD_NOISE = re.compile(
        r"^(remind me to|remind me|add (a )?task( to)?|create (a )?task( to)?|new task"
        r"|i need to|i have to|remember to|schedule|add a reminder( to)?|"
        r"add reminder( to)?)\s*",
        re.IGNORECASE,
    )

    # Trailing date noise
    _DATE_SUFFIX = re.compile(
        r"\s*(by|on|before|due)\s+(today|tomorrow|next week|\d{4}-\d{2}-\d{2})$",
        re.IGNORECASE,
    )

    # Words to strip when pulling a task name from a COMPLETE sentence
    # e.g. "okay, i've completed add cooking task" → "add cooking"
    _COMPLETE_NOISE = re.compile(
        r"\b(okay|ok|so|yes|yep|i've|i have|i|completed|complete|"
        r"done|finished|finish|mark|as done|the|task)\b",
        re.IGNORECASE,
    )

    # Words to strip when pulling a task name from a DELETE sentence
    _DELETE_NOISE = re.compile(
        r"\b(okay|ok|please|delete|remove|cancel|erase|the|task|from( the)? list)\b",
        re.IGNORECASE,
    )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_title(self, text: str) -> str | None:
        """
        Strip ADD intent noise and return the cleaned title.
        Returns None if nothing useful remains.
        """
        cleaned = self._ADD_NOISE.sub("", text).strip()
        cleaned = self._DATE_SUFFIX.sub("", cleaned).strip()
        # If what's left is very short or is still an intent phrase, skip it
        if len(cleaned) < 2:
            return None
        return cleaned.capitalize()

    def extract_task_name(self, text: str) -> str | None:
        """
        Pull a task name from a COMPLETE or DELETE sentence by stripping
        the action verb noise words.
        Returns None if nothing useful remains.
        """
        # Try complete noise first
        for pattern in (self._COMPLETE_NOISE, self._DELETE_NOISE):
            cleaned = pattern.sub(" ", text).strip(" ,.")
            # collapse multiple spaces
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            if len(cleaned) > 1:
                return cleaned
        return None

    def extract_date(self, text: str) -> str | None:
        t = text.lower()
        today = datetime.today()

        if "today" in t:
            return today.strftime("%Y-%m-%d")
        if "tomorrow" in t:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if "next week" in t:
            return (today + timedelta(days=7)).strftime("%Y-%m-%d")

        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if match:
            return match.group(1)
        return None

    def extract_category(self, text: str) -> str:
        categories = {
            "work":     ["work", "job", "meeting", "office", "project", "client"],
            "health":   ["gym", "health", "doctor", "exercise", "workout", "run"],
            "personal": ["home", "family", "personal", "friend", "errand"],
        }
        t = text.lower()
        for cat, keywords in categories.items():
            if any(k in t for k in keywords):
                return cat
        return "personal"

    def extract_task_reference(self, text: str) -> str | None:
        """Return a bare numeric task ID if one appears in the text."""
        match = re.search(r"\b(\d+)\b", text)
        return match.group(1) if match else None

    def extract_delete_reference(self, text: str) -> str | None:
        """Return task ID when user says 'delete 3' or 'remove task 5'."""
        match = re.search(
            r"(?:delete|remove)\s+(?:task\s+)?(\d+)", text, re.IGNORECASE
        )
        return match.group(1) if match else None