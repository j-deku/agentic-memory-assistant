# paste this into a new file called inspect_data.py and run it
import json
from pathlib import Path
from collections import Counter

path = Path("training_data.jsonl")
examples = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

tools = Counter()
issues = []

for i, ex in enumerate(examples):
    convs = ex.get("conversations", [])
    if len(convs) != 3:
        issues.append(f"Line {i}: expected 3 turns, got {len(convs)}")
        continue

    assistant_turn = convs[2]["content"]
    try:
        parsed = json.loads(assistant_turn)
        tool = parsed.get("name", "MISSING")
        tools[tool] += 1

        if tool == "set_reminder":
            args = parsed.get("arguments", {})
            if not args.get("task_description") or not args.get("time_str"):
                issues.append(f"Line {i}: set_reminder missing args — {args}")

    except json.JSONDecodeError:
        issues.append(f"Line {i}: assistant turn is not valid JSON — {assistant_turn[:80]}")

print(f"Total examples : {len(examples)}")
print(f"\nTool distribution:")
for tool, count in tools.most_common():
    print(f"  {tool:35s} {count:4d}  ({100*count/len(examples):.1f}%)")

print(f"\nIssues found   : {len(issues)}")
for issue in issues[:10]:
    print(f"  {issue}")

print(f"\nSample (line 0) user input:")
print(f"  {examples[0]['conversations'][1]['content']}")
print(f"Sample (line 0) assistant output:")
print(f"  {examples[0]['conversations'][2]['content']}")