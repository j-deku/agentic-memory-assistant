"""
test_intent_llm.py

Run this on your machine (with Ollama running) to confirm intent_llm.py
correctly classifies intents against a live qwen2.5:1.5b before wiring it
into dialogue_manager.py.

Usage:
    python test_intent_llm.py
"""

import time
from intent_llm import classify_intent

FAKE_TASKS = [
    {"id": 1, "title": "Go to campus", "completed": 0},
    {"id": 2, "title": "Call sisters", "completed": 0},
    {"id": 3, "title": "Finish report", "completed": 0},
    {"id": 4, "title": "Buy groceries", "completed": 1},
]

TEST_CASES = [
    "go to campus",
    "i wrapped up the report",
    "call sisters",
    "remind me to pay rent tomorrow",
    "show my tasks",
    "thanks!",
    "delete task 3",
    "what should i do today",
]

print(f"Testing against {len(TEST_CASES)} phrases...\n")

for text in TEST_CASES:
    start = time.time()
    result = classify_intent(text, FAKE_TASKS, user_name="Jeremiah")
    elapsed = time.time() - start

    print(f'You said: "{text}"')
    if "_error" in result:
        print(f"  ERROR: {result['_error']}")
    else:
        print(f"  intent={result['intent']}  task_id={result['task_id']}  "
              f"title={result['title']}  confidence={result['confidence']}")
    print(f"  ({elapsed:.2f}s)\n")