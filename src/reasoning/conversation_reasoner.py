import re

COMPLETE_PATTERNS = [
    r"\bi completed\b",
    r"\bi've completed\b",
    r"\bcompleted\b",
    r"\bfinished\b",
    r"\bdone\b",
    r"\balready did\b",
]

DELETE_PATTERNS = [
    r"\bdelete\b",
    r"\bremove\b",
    r"\bcancel\b",
]

ADD_PATTERNS = [
    r"\badd\b",
    r"\bremind\b",
    r"\bi need to\b",
    r"\bi have to\b",
]

def detect_override(text, current_intent, awaiting_slot):

    if not awaiting_slot:
        return None

    lower = text.lower()

    for pattern in COMPLETE_PATTERNS:
        if re.search(pattern, lower):
            return {
                "intent": "complete_task",
                "reason": "user_completed_task"
            }

    for pattern in DELETE_PATTERNS:
        if re.search(pattern, lower):
            return {
                "intent": "delete_task",
                "reason": "user_delete_request"
            }

    for pattern in ADD_PATTERNS:
        if re.search(pattern, lower):
            return {
                "intent": "add_task",
                "reason": "user_new_task"
            }

    return None