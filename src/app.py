"""
app.py  —  Unified entry point

Merges assistant.py (LangGraph + reminders + search) with main.py
(DialogueManager + task CRUD + habits + productivity).

Architecture:
  SmartRouter classifies each input, then routes to:
    - DialogueManager  → task CRUD, habits, productivity  → template response
    - LangGraph agent  → reminders, scheduling, search, fallback → LLM response

Neither assistant.py nor main.py is modified. This file is the only
new code needed to combine both systems.

Usage:
  python app.py
"""

import os
import sys
import traceback

# ---------------------------------------------------------------------------
# 1. BOOTSTRAP  (from main.py — run once at startup)
# ---------------------------------------------------------------------------
from agentic.agent_logger import create_agent_events_table
from context_handler import handle_context
from memory_based_greeting import memory_greeting
from nlp.pipeline import NLUPipeline
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

create_agent_events_table()
create_tasks_table()
create_memory_table()
create_learning_events_table()
create_prediction_table()
init_rl_memory()
init_conversation_memory()

startup_message()
memory_greeting()

# ---------------------------------------------------------------------------
# 2. TASK FUNCTION REGISTRY  (unchanged from main.py)
# ---------------------------------------------------------------------------
_task_fns = {
    "add_task":                      add_task,
    "list_tasks":                    list_tasks,
    "complete_task":                 complete_task,
    "delete_task":                   delete_task,
    "get_overdue_tasks":             get_overdue_tasks,
    "get_recommended_tasks":         get_recommended_tasks,
    "get_task_score":                get_task_score,
    "analyze_habits":                analyze_habits,
    "normalize_date":                normalize_date,
    "productivity_score":            productivity_score,
    "get_memory":                    get_memory,
    "update_memory_from_tasks":      update_memory_from_tasks,
    "predictive_insights":           predictive_insights,
    "generate_briefing":             generate_briefing,
    "update_prediction":             update_prediction,
    "log_event":                     log_event,
    "self_learning_cycle":           self_learning_cycle,
    "retrain_if_needed":             retrain_if_needed,
    "retrain_transformer_if_needed": retrain_transformer_if_needed,
    "update_rl_memory":              update_rl_memory,
}

# ---------------------------------------------------------------------------
# 3. DIALOGUE MANAGER  (task CRUD layer — from main.py)
# ---------------------------------------------------------------------------
from dialogue_manager import DialogueManager
from agentic.orchestrator import AgentOrchestrator
from agentic.agent_logger import log_agent_event

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

# ---------------------------------------------------------------------------
# 4. LANGGRAPH AGENT  (reminders + search layer — from assistant.py)
# ---------------------------------------------------------------------------
import logging
import pytz
from datetime import datetime
from typing import Annotated, TypedDict, List, Optional

import dateparser
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ── Timezone ────────────────────────────────────────────────────────────────
USER_TIMEZONE = "Africa/Accra"
LOCAL_TZ = pytz.timezone(USER_TIMEZONE)

def _now_local() -> datetime:
    return datetime.now(LOCAL_TZ)

def _parse_local_time(time_str: str) -> Optional[datetime]:
    parsed = dateparser.parse(
        time_str,
        settings={
            "PREFER_DATES_FROM": "current_period",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": USER_TIMEZONE,
            "RELATIVE_BASE": datetime.now(LOCAL_TZ).replace(tzinfo=None),
        },
    )
    if parsed is None:
        return None
    return parsed.astimezone(LOCAL_TZ) if parsed.tzinfo else LOCAL_TZ.localize(parsed)

# ── Scheduler ────────────────────────────────────────────────────────────────
_logger = logging.getLogger("app")

def _reminder_fired(task_description: str):
    ts = _now_local().strftime("%Y-%m-%d %H:%M %Z")
    _logger.info(f"REMINDER FIRED at {ts}: {task_description}")
    print(f"\n⏰ REMINDER [{ts}]: {task_description}\n")

def _scheduler_listener(event):
    if event.exception:
        _logger.error(f"Scheduled job failed: {event.job_id} — {event.exception}")

_scheduler = BackgroundScheduler(timezone=LOCAL_TZ)
_scheduler.add_listener(_scheduler_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
_scheduler.start()
_reminders: list = []

# ── LangGraph tools ──────────────────────────────────────────────────────────
@tool
def set_reminder(task_description: str, time_str: str) -> str:
    """
    Sets a one-time or recurring reminder.

    One-time examples:  'today at 3pm', 'tomorrow at 10am', 'in 2 hours'
    Recurring examples: 'every day at 8am', 'every Monday at 9am'
    """
    try:
        time_lower = time_str.lower().strip()
        is_recurring = time_lower.startswith("every")

        if is_recurring:
            job = _schedule_recurring(task_description, time_lower)
            label = _describe_recurring(time_lower)
            _reminders.append({
                "task": task_description, "time_str": time_str,
                "run_time": label, "job_id": job.id, "recurring": True,
            })
            return f"Confirmed: I'll remind you to '{task_description}' {label} ({USER_TIMEZONE})."

        run_time = _parse_local_time(time_str)
        if run_time is None:
            raise ValueError(f"Could not understand the time: '{time_str}'")
        if run_time < _now_local():
            raise ValueError(f"'{run_time.strftime('%A, %b %d at %I:%M %p %Z')}' is in the past.")

        job = _scheduler.add_job(
            _reminder_fired, trigger="date", run_date=run_time,
            args=[task_description], id=f"reminder_{len(_reminders)}", timezone=LOCAL_TZ,
        )
        _reminders.append({
            "task": task_description, "time_str": time_str,
            "run_time": run_time.strftime("%Y-%m-%d %H:%M %Z"),
            "job_id": job.id, "recurring": False,
        })
        return f"Confirmed: I'll remind you to '{task_description}' on {run_time.strftime('%A, %b %d at %I:%M %p')} ({USER_TIMEZONE})."

    except ValueError as e:
        return f"Sorry, I couldn't set that reminder: {e}"
    except Exception as e:
        _logger.error(f"set_reminder error: {e}", exc_info=True)
        return "Sorry, an unexpected error occurred while setting your reminder."


def _schedule_recurring(task_description: str, time_lower: str):
    DAY_MAP = {
        "monday": "mon", "tuesday": "tue", "wednesday": "wed",
        "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun",
        "weekday": "mon-fri", "weekend": "sat,sun", "day": "*",
    }
    time_part = time_lower.replace("every", "").strip()
    for day_key in DAY_MAP:
        time_part = time_part.replace(day_key, "").strip()
    time_part = time_part.replace("at", "").strip()
    parsed_time = dateparser.parse(time_part or "now", settings={"RETURN_AS_TIMEZONE_AWARE": False})
    hour = parsed_time.hour if parsed_time else 9
    minute = parsed_time.minute if parsed_time else 0
    day_of_week = "*"
    for key, cron_val in DAY_MAP.items():
        if key in time_lower:
            day_of_week = cron_val
            break
    trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute, timezone=LOCAL_TZ)
    return _scheduler.add_job(
        _reminder_fired, trigger=trigger, args=[task_description], id=f"reminder_{len(_reminders)}"
    )


def _describe_recurring(time_lower: str) -> str:
    if "weekday" in time_lower:
        return "every weekday (Mon–Fri)"
    if "weekend" in time_lower:
        return "every weekend (Sat–Sun)"
    if "day" in time_lower and "monday" not in time_lower and "wednesday" not in time_lower:
        return "every day"
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in time_lower:
            return f"every {day.capitalize()}"
    return "on a recurring schedule"


@tool
def get_upcoming_reminders() -> str:
    """Retrieves all scheduled reminders."""
    if not _reminders:
        return "No reminders scheduled."
    lines = []
    for r in _reminders:
        label = "🔁 Recurring" if r.get("recurring") else "🔔 One-time"
        lines.append(f"{label} | {r['task']} → {r['run_time']}")
    return f"Scheduled reminders ({USER_TIMEZONE}):\n" + "\n".join(lines)


@tool
def open_and_search(query: str) -> str:
    """Opens a browser, searches DuckDuckGo, and returns the top result."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            try:
                page = browser.new_page()
                page.goto(f"https://duckduckgo.com/?q={query}&ia=web", timeout=15000)
                page.wait_for_selector("[data-testid='result']", timeout=10000)
                result = page.inner_text("[data-testid='result']")[:500]
                return f"Search result for '{query}': {result}..."
            except PlaywrightTimeout:
                return "Sorry, the search timed out. Please try again."
            except Exception as e:
                _logger.error(f"Browser error: {e}", exc_info=True)
                return "Sorry, an error occurred during the browser search."
            finally:
                browser.close()
    except Exception as e:
        _logger.error(f"Playwright launch failed: {e}", exc_info=True)
        return "Sorry, the browser could not be launched."


# ── LangGraph agent ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "conversation history"]

_agent_tools = [set_reminder, get_upcoming_reminders, open_and_search]
_tool_node = ToolNode(_agent_tools)

# Swap "qwen2.5:1.5b" for your fine-tuned model name once ready
_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0).bind_tools(_agent_tools)

_AGENT_SYSTEM = f"""You are a professional, friendly personal assistant named Aria.
Today's date/time: {_now_local().strftime('%A, %B %d %Y %I:%M %p')} ({USER_TIMEZONE}).

You can set reminders, check upcoming reminders, and search the web.
For task management the user's task system handles that — you don't need to manage tasks yourself.

Always be concise, warm, and professional. When you confirm a reminder or search result,
give a clear, natural response — not just a raw tool output."""

def _call_model(state: AgentState):
    messages = [SystemMessage(content=_AGENT_SYSTEM)] + state["messages"]
    response = _llm.invoke(messages)
    return {"messages": [response]}

def _should_continue(state: AgentState):
    return "tools" if state["messages"][-1].tool_calls else END

_workflow = StateGraph(AgentState)
_workflow.add_node("agent", _call_model)
_workflow.add_node("tools", _tool_node)
_workflow.set_entry_point("agent")
_workflow.add_conditional_edges("agent", _should_continue)
_workflow.add_edge("tools", "agent")
_langgraph_app = _workflow.compile()

def _run_langgraph_agent(user_input: str) -> str:
    """Run the LangGraph agent and return its final text response."""
    try:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        final_response = ""
        for output in _langgraph_app.stream(inputs):
            for key, value in output.items():
                if key == "agent":
                    content = value["messages"][-1].content
                    if content:
                        final_response = content
        return final_response or "I couldn't process that. Please try again."
    except Exception as e:
        _logger.error(f"LangGraph agent error: {e}", exc_info=True)
        return "Sorry, something went wrong with the assistant."

# ---------------------------------------------------------------------------
# 5. SMART ROUTER
# ---------------------------------------------------------------------------
# Intents that belong to DialogueManager — matches Goal enum from NLUPipeline
from brain.goal_types import Goal

_TASK_INTENTS = {
    Goal.ADD_TASK, Goal.LIST_TASKS, Goal.COMPLETE_TASK, Goal.DELETE_TASK,
    Goal.GET_OVERDUE_TASKS, Goal.PRODUCTIVITY_SCORE, Goal.GET_RECOMMENDED_TASKS,
    Goal.ANALYZE_HABITS, Goal.GREETING, Goal.ACKNOWLEDGEMENT, Goal.GET_USER_NAME,
    "add_task", "list_tasks", "complete_task", "delete_task",
    "get_overdue_tasks", "productivity_score", "get_recommended_tasks",
    "analyze_habits", "greeting", "acknowledgement", "get_user_name",
}

# Keywords that signal reminder/search intent — route to LangGraph
_AGENT_KEYWORDS = (
    "remind", "reminder", "schedule", "alarm", "alert", "every day",
    "every week", "every monday", "every tuesday", "every wednesday",
    "every thursday", "every friday", "every saturday", "every sunday",
    "search", "look up", "find online", "google", "browse",
    "what is", "who is", "how do i", "how does",
)

# Keywords that strongly signal task management — route to DialogueManager
_TASK_KEYWORDS = (
    "add task", "create task", "new task", "add a task", "add to",
    "show tasks", "list tasks", "my tasks", "show my tasks", "view tasks",
    "complete task", "mark done", "mark complete", "finish task",
    "delete task", "remove task",
    "overdue", "overdue tasks",
    "productivity", "productivity score",
    "recommend", "what should i do", "suggest",
    "habits", "analyze habits",
)

def _smart_route(user_input: str, parsed_intent: str) -> str:
    """
    Returns 'task' or 'agent'.

    Priority:
      1. If the NLU produced a known task intent -> task
      2. If input contains task keywords -> task
      3. If input contains reminder/search keywords -> agent
      4. Fallback -> agent
    """
    if parsed_intent in _TASK_INTENTS:
        return "task"

    lower = user_input.lower()

    if any(kw in lower for kw in _TASK_KEYWORDS):
        return "task"

    if any(kw in lower for kw in _AGENT_KEYWORDS):
        return "agent"

    return "agent"

# ---------------------------------------------------------------------------
# 6. UNIFIED PROCESS FUNCTION
# ---------------------------------------------------------------------------
def process(user_input: str) -> str:
    """
    Single function that handles any user input.
    Routes to the right system and returns a response string.
    """
    # Layer 1: memory brain (unchanged from main.py)
    memory_response = learn_from_text(user_input)
    if memory_response:
        dm.state.add_turn("user", user_input)
        dm.state.add_turn("assistant", memory_response)
        return memory_response

    memory_answer = answer_memory_question(user_input)
    if memory_answer:
        dm.state.add_turn("user", user_input)
        dm.state.add_turn("assistant", memory_answer)
        return memory_answer

    # Layer 2: if we're mid-conversation in DialogueManager (slot filling),
    # always honour that context first
    if dm.state.awaiting_slot:
        return orchestrator.run(user_input)

    # Layer 3: parse intent and route
    parsed = engine.parse(user_input)
    intent = parsed.intent or ""
    route = _smart_route(user_input, intent)

    if route == "task":
        return orchestrator.run(user_input)
    else:
        return _run_langgraph_agent(user_input)

# ---------------------------------------------------------------------------
# 7. MAIN LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nAssistant ready. Type naturally — or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
            print("Assistant: Goodbye! Your tasks are saved.")
            break

        DEBUG = False
        if DEBUG:
            result = engine.parse(user_input)
            route = _smart_route(user_input, result.intent or "")
            print(f"[DEBUG] intent={result.intent} route={route} title={result.title} due={result.due_date}")

        try:
            response = process(user_input)
            print(f"Assistant: {response}")
        except Exception as e:
            _logger.error(f"process() failed: {e}", exc_info=True)
            print("Assistant: Sorry, something went wrong. Please try again.")