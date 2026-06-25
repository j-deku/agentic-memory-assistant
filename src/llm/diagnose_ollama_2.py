"""
diagnose_ollama_2.py

Isolates whether it's the system prompt length, the schema size, or
something else causing the timeout. Run on your machine, no Ollama
restart needed.
"""

import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"


def call(label, payload, timeout):
    print(f"{label}")
    print("-" * 60)
    start = time.time()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        content = body.get("message", {}).get("content", "")
        print(f"  SUCCESS in {elapsed:.2f}s")
        print(f"  Response: {content!r}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAILED after {elapsed:.2f}s: {e}")
    print()


# TEST A: short system prompt + small schema (like the one that worked before)
small_schema = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["add_task", "chat"]},
        "task_id": {"type": ["integer", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "task_id", "confidence"],
}
call("TEST A: small schema, no system prompt, 20s timeout", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "classify: go to campus"}],
    "format": small_schema,
    "stream": False,
    "keep_alive": "30m",
}, 20)

# TEST B: small schema WITH a system prompt added
call("TEST B: small schema + short system prompt, 20s timeout", {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You classify intents. Return JSON only."},
        {"role": "user", "content": "classify: go to campus"},
    ],
    "format": small_schema,
    "stream": False,
    "keep_alive": "30m",
}, 20)

# TEST C: full 6-field schema, no system prompt
full_schema_no_enum = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "task_id": {"type": ["integer", "null"]},
        "title": {"type": ["string", "null"]},
        "due_date": {"type": ["string", "null"]},
        "category": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "task_id", "title", "due_date", "category", "confidence"],
}
call("TEST C: full 6-field schema (no enum), no system prompt, 25s timeout", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "classify: go to campus"}],
    "format": full_schema_no_enum,
    "stream": False,
    "keep_alive": "30m",
}, 25)

# TEST D: full 6-field schema WITH the enum constraint on intent
full_schema_with_enum = dict(full_schema_no_enum)
full_schema_with_enum["properties"] = dict(full_schema_no_enum["properties"])
full_schema_with_enum["properties"]["intent"] = {
    "type": "string",
    "enum": [
        "add_task", "complete_task", "delete_task", "view_tasks",
        "overdue", "recommend", "productivity", "habits", "briefing",
        "memory", "temporal_query", "goodbye", "chat"
    ]
}
call("TEST D: full 6-field schema WITH 13-value enum, no system prompt, 25s timeout", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "classify: go to campus"}],
    "format": full_schema_with_enum,
    "stream": False,
    "keep_alive": "30m",
}, 25)