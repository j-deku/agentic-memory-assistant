"""
main.py  (conversational rewrite)

The routing loop has been replaced by DialogueManager.
All existing imports and DB/ML setup code is unchanged.
"""

import os
import traceback
from adaptive_response_engine import personalize_response
from agentic.agent_logger import create_agent_events_table
from context_handler import handle_context
from memory_based_greeting import memory_greeting
from nlp.pipeline import NLUPipeline
from nlp.predict_torch_intent import predict_intent
from tasks import (
    create_tasks_table,
    create_memory_table,
    add_task,
    delete_task,
    list_tasks,
    complete_task,
    get_overdue_tasks,
    get_recommended_tasks,
    get_task_score,
    analyze_habits,
    normalize_date,
    productivity_score,
    get_memory,
    update_memory_from_tasks,
    predictive_insights,
    generate_briefing,
    create_learning_events_table,
    create_prediction_table,
    update_prediction,
)

from auto_trainer import retrain_if_needed
from learning_logger import log_event
from self_learning_loop import self_learning_cycle
from rl_memory import init_rl_memory
from transformer_auto_trainer import retrain_transformer_if_needed
from startup_assistant import startup_message
from conversation_memory_db import init_conversation_memory
from memory_brain import learn_from_text, answer_memory_question

try:
    from rl_memory import update_rl_memory
except ImportError:
    update_rl_memory = None

# ─────────────────────────────────────────────
# SETUP  (unchanged from original)
# ─────────────────────────────────────────────
create_agent_events_table()
create_tasks_table()
create_memory_table()
create_learning_events_table()
create_prediction_table()
init_rl_memory()
init_conversation_memory()

startup_message()
memory_greeting()

# ─────────────────────────────────────────────
# WIRE UP DIALOGUE MANAGER
# ─────────────────────────────────────────────
# Pass all your existing functions in so DialogueManager
# can call them without importing them directly.
# Add/remove entries here freely as your project grows.

from dialogue_manager import DialogueManager
from agentic.agent_logger import log_agent_event


_task_fns = {
    "add_task":                   add_task,
    "list_tasks":                 list_tasks,
    "complete_task":              complete_task,
    "delete_task":                delete_task,
    "get_overdue_tasks":          get_overdue_tasks,
    "get_recommended_tasks":      get_recommended_tasks,
    "get_task_score":             get_task_score,
    "analyze_habits":             analyze_habits,
    "normalize_date":             normalize_date,
    "productivity_score":         productivity_score,
    "get_memory":                 get_memory,
    "update_memory_from_tasks":   update_memory_from_tasks,
    "predictive_insights":        predictive_insights,
    "generate_briefing":          generate_briefing,
    "update_prediction":          update_prediction,
    "log_event":                  log_event,
    "self_learning_cycle":        self_learning_cycle,
    "retrain_if_needed":          retrain_if_needed,
    "retrain_transformer_if_needed": retrain_transformer_if_needed,
    "update_rl_memory":           update_rl_memory,
}

from agentic.orchestrator import AgentOrchestrator          

engine = NLUPipeline()

dm = DialogueManager(
    _task_fns,
    user_name="Jeremiah",
    reasoning=engine,
)

orchestrator = AgentOrchestrator(            
    nlu=engine,
    dialogue_mgr=dm,
    task_fns=_task_fns,
    log_event=_task_fns.get("log_agent_event"),
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
    DEBUG = False

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

    response = orchestrator.run(user_input)   
    print(f"Assistant: {response}")