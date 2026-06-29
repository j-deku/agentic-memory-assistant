# paste in a quick test file
from nlp.pipeline import NLUPipeline
engine = NLUPipeline()

tests = [
    "Add task to call vidash today",
    "show my overdue tasks",
    "remove the call vidash task from the list",
    "show tasks",
    "what should i do today?",
    "what's my productivity score?"
]
for t in tests:
    result = engine.parse(t)
    print(f"'{t}' → intent={result.intent} title={result.title}")