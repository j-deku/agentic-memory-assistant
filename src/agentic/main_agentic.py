"""
main.py  (agentic rewrite — diff from conversational version)

Only the WIRE-UP and MAIN LOOP sections change.
Everything above the separator (DB setup, imports, ML init) is identical
to the previous version and is not repeated here.

──────────────────────────────────────────────────
WHAT CHANGED vs the conversational main.py
──────────────────────────────────────────────────

  BEFORE:
    dm = DialogueManager(...)
    ...
    response = dm.process(user_input)

  AFTER:
    dm = DialogueManager(...)        ← unchanged
    orchestrator = AgentOrchestrator(engine, dm, _task_fns, log_event)
    ...
    response = orchestrator.run(user_input)   ← new entry point

DialogueManager is still instantiated; AgentOrchestrator wraps it.
Slot-filling flows (add-task multi-turn, etc.) are transparently
delegated back to DialogueManager when dm.state.awaiting_slot is set.
──────────────────────────────────────────────────
"""

# ── All original imports and DB/ML setup is UNCHANGED above this line ──

# ─────────────────────────────────────────────
# WIRE UP  (replaces the previous wiring block)
# ─────────────────────────────────────────────

from dialogue_manager import DialogueManager
from agent import AgentOrchestrator          # NEW

engine = NLUPipeline()

dm = DialogueManager(
    _task_fns,
    user_name="Jeremiah",
    reasoning=engine,
)

orchestrator = AgentOrchestrator(            # NEW
    nlu=engine,
    dialogue_mgr=dm,
    task_fns=_task_fns,
    log_event=_task_fns.get("log_event"),
)

# ─────────────────────────────────────────────
# MAIN LOOP  (one line changes)
# ─────────────────────────────────────────────

print("\nAssistant ready. Type naturally — or 'exit' to quit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye! 👋")
        break

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
        print("Assistant: Goodbye! Your tasks are saved. 👋")
        break

    # ── Layer 1: memory brain (unchanged) ───────────────────────────────
    memory_response = learn_from_text(user_input)
    if memory_response:
        print(f"Assistant: {memory_response}")
        dm.state.add_turn("user", user_input)
        dm.state.add_turn("assistant", memory_response)
        continue

    memory_answer = answer_memory_question(user_input)
    if memory_answer:
        print(f"Assistant: {memory_answer}")
        dm.state.add_turn("user", user_input)
        dm.state.add_turn("assistant", memory_answer)
        continue

    # ── Layer 2: AgentOrchestrator  (was: dm.process) ───────────────────
    if DEBUG:
        reasoning = engine.analyze(user_input)
        print(reasoning)
        result = engine.parse(user_input)
        print(
            f"[DEBUG] "
            f"intent={result.intent} "
            f"title={result.title} "
            f"due={result.due_date} "
            f"category={result.category}"
        )

    response = orchestrator.run(user_input)   # ← was dm.process(user_input)
    print(f"Assistant: {response}")