"""
diagnose_ollama.py

Minimal diagnostic to isolate where the timeout is coming from.
Run this on your machine: python diagnose_ollama.py
"""

import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

print("TEST 1: Plain chat, no format constraint, short timeout (10s)")
print("-" * 60)
payload1 = {
    "model": "qwen2.5:1.5b",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "stream": False,
}
try:
    start = time.time()
    data = json.dumps(payload1).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print(f"  SUCCESS in {time.time()-start:.2f}s")
    print(f"  Response: {body.get('message', {}).get('content', '')!r}")
except Exception as e:
    print(f"  FAILED after {time.time()-start:.2f}s: {e}")

print()
print("TEST 2: format='json' (simple JSON mode, no schema), 15s timeout")
print("-" * 60)
payload2 = {
    "model": "qwen2.5:1.5b",
    "messages": [{"role": "user", "content": "Return JSON with a single key 'greeting' set to 'hello'"}],
    "format": "json",
    "stream": False,
}
try:
    start = time.time()
    data = json.dumps(payload2).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print(f"  SUCCESS in {time.time()-start:.2f}s")
    print(f"  Response: {body.get('message', {}).get('content', '')!r}")
except Exception as e:
    print(f"  FAILED after {time.time()-start:.2f}s: {e}")

print()
print("TEST 3: Full JSON schema (the real intent_llm.py schema), 30s timeout")
print("-" * 60)
schema = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["add_task", "chat"]},
        "task_id": {"type": ["integer", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "task_id", "confidence"],
}
payload3 = {
    "model": "qwen2.5:1.5b",
    "messages": [{"role": "user", "content": "classify: hello there"}],
    "format": schema,
    "stream": False,
}
try:
    start = time.time()
    data = json.dumps(payload3).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print(f"  SUCCESS in {time.time()-start:.2f}s")
    print(f"  Response: {body.get('message', {}).get('content', '')!r}")
except Exception as e:
    print(f"  FAILED after {time.time()-start:.2f}s: {e}")