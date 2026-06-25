"""
intent_llm.py

Replaces regex-based intent detection with a single structured call to a
local Ollama model. This is what makes Aerial actually understand meaning
("go to campus" matching the existing task "Go to campus") instead of
pattern-matching against a hand-written verb list.

Requires Ollama running locally (default http://localhost:11434) with
qwen2.5:1.5b pulled. No API key, no internet — fully local and private.

Uses Ollama's `format` parameter with a JSON schema, which constrains the
model's decoding so the output is ALWAYS valid JSON matching this exact
shape. No markdown fences to strip, no "Sure, here's the JSON!" preamble,
no parse failures to catch — the schema is enforced at generation time.
"""

import json
import urllib.request
import urllib.error
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"

_SYSTEM_PROMPT = """You are the intent-parsing layer for a personal task assistant called Aerial.
Given the user's message and their current task list, classify what the user wants.

Rules:
- If the message refers to something semantically close to an existing task title
  (even worded completely differently — "go to campus" matches "Go to campus",
  "wrap up the report" matches "Finish report"), set task_id to that task's id.
  Usually the intent is complete_task when they're describing something as done.
- If nothing in the task list matches and the message describes a new actionable
  thing, use add_task and extract a clean, capitalized title.
- If the message is plain conversation (greetings, thanks, small talk, questions
  about yourself), use intent "chat" and leave the other fields null.
- Never invent a task_id that isn't in the provided list.
- Set confidence below 0.6 if genuinely unsure between two intents.
"""

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "add_task", "complete_task", "delete_task", "view_tasks",
                "overdue", "recommend", "productivity", "habits", "briefing",
                "memory", "temporal_query", "goodbye", "chat"
            ]
        },
        "task_id": {"type": ["integer", "null"]},
        "title": {"type": ["string", "null"]},
        "due_date": {"type": ["string", "null"]},
        "category": {"type": ["string", "null"]},
        "confidence": {"type": "number"}
    },
    "required": ["intent", "task_id", "title", "due_date", "category", "confidence"]
}

_FALLBACK_RESULT = {
    "intent": "chat",
    "task_id": None,
    "title": None,
    "due_date": None,
    "category": None,
    "confidence": 0.0,
}


def preload_model(timeout: float = 60.0) -> bool:
    """
    Loads the model into memory and pins it there for 30 minutes.
    Call this once when Aerial starts up (in main.py) so the user's
    first real message doesn't pay the 5-30s cold-start cost.

    Returns True if the model loaded successfully, False otherwise.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": ""}],
        "stream": False,
        "keep_alive": "30m",
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True
    except Exception:
        return False


def classify_intent(user_text: str, tasks: list, user_name: str = "", timeout: float = 25.0) -> dict:
    """
    Calls the local Ollama model to classify intent + extract slots in one shot.

    `tasks` should be a list of dicts or tuples compatible with your existing
    task store, e.g. [{"id": 1, "title": "...", "completed": 0}, ...]

    Returns a dict matching _JSON_SCHEMA's shape. On any failure (Ollama not
    running, timeout, bad response) returns a safe "chat" fallback so the
    caller can route to the existing conversational layer instead of crashing.
    """
    task_lines = []
    for t in tasks:
        if isinstance(t, dict):
            tid = t.get("id")
            title = t.get("title", "")
            completed = bool(t.get("completed"))
        else:
            tid = t[0] if len(t) > 0 else None
            title = t[1] if len(t) > 1 else ""
            completed = bool(t[4]) if len(t) > 4 else False
        task_lines.append(f'  id={tid}: "{title}" (completed={completed})')

    task_summary = "\n".join(task_lines) if task_lines else "  (no tasks yet)"

    user_prompt = (
        f"User: {user_name or 'the user'}\n"
        f"Current tasks:\n{task_summary}\n\n"
        f'Message: "{user_text}"'
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "format": _JSON_SCHEMA,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1,
        },
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body.get("message", {}).get("content", "")
        parsed = json.loads(content)

        result = dict(_FALLBACK_RESULT)
        result.update(parsed)
        return result

    except urllib.error.URLError as e:
        result = dict(_FALLBACK_RESULT)
        result["_error"] = f"Ollama unreachable: {e}"
        return result

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        result = dict(_FALLBACK_RESULT)
        result["_error"] = f"Bad response from model: {e}"
        return result

    except Exception as e:
        result = dict(_FALLBACK_RESULT)
        result["_error"] = f"Unexpected error: {e}"
        return result