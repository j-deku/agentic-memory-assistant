# nlp/task_extractor.py

import re

COMPLETE_PATTERNS = [

    r"mark\s+(.*?)\s+as\s+completed",
    r"mark\s+(.*?)\s+completed",
    r"mark\s+(.*?)\s+complete",

    r"complete\s+(.*)",
    r"completed\s+(.*)",

    r"finish\s+(.*)",
    r"finished\s+(.*)",

    r"done\s+(.*)",

    r"i've completed\s+(.*)",
    r"i completed\s+(.*)",

    r"already completed\s+(.*)",
]

def extract_task_reference(text):

    text = text.strip()

    for pattern in COMPLETE_PATTERNS:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    return None