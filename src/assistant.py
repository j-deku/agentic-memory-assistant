import logging
import sys
import pytz
from datetime import datetime
from typing import Annotated, TypedDict, List, Optional

import dateparser
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from langchain_core.messages import SystemMessage

from prompts.system_prompt import system_prompt


# ---------------------------------------------------------------------------
# 1. LOGGING
# ---------------------------------------------------------------------------

# File handler — captures everything including third-party logs
file_handler = logging.FileHandler("assistant.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

# Console handler — only shows logs from THIS app
class AppOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("assistant")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
console_handler.addFilter(AppOnlyFilter())

# Root logger: all libraries write to file, only app logs hit console
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger("assistant")
# ---------------------------------------------------------------------------
# 2. TIMEZONE CONFIG — change this to your local timezone
#    Full list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
#    Examples: "Africa/Accra", "America/New_York", "Europe/London", "Asia/Lagos"
# ---------------------------------------------------------------------------

USER_TIMEZONE = "Africa/Accra"
LOCAL_TZ = pytz.timezone(USER_TIMEZONE)

def now_local() -> datetime:
    """Returns the current time in the user's local timezone (timezone-aware)."""
    return datetime.now(LOCAL_TZ)

def parse_local_time(time_str: str) -> Optional[datetime]:
    """
    Parses a natural language time string into a timezone-aware datetime
    anchored to the user's local timezone.
    """
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
    # Ensure result is always in the user's local timezone
    if parsed.tzinfo is None:
        parsed = LOCAL_TZ.localize(parsed)
    else:
        parsed = parsed.astimezone(LOCAL_TZ)
    return parsed


# ---------------------------------------------------------------------------
# 3. SCHEDULER
# ---------------------------------------------------------------------------

def _reminder_fired(task_description: str):
    """Callback that runs when a scheduled reminder triggers."""
    local_now = now_local().strftime("%Y-%m-%d %H:%M %Z")
    logger.info(f"🔔 REMINDER FIRED at {local_now}: {task_description}")
    print(f"\n⏰ REMINDER [{local_now}]: {task_description}\n")


def _scheduler_listener(event):
    if event.exception:
        logger.error(f"Scheduled job failed: {event.job_id} — {event.exception}")
    else:
        logger.info(f"Scheduled job completed: {event.job_id}")


_scheduler = BackgroundScheduler(timezone=LOCAL_TZ)
_scheduler.add_listener(_scheduler_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
_scheduler.start()
_tasks: list = []


# ---------------------------------------------------------------------------
# 4. TOOLS
# ---------------------------------------------------------------------------

@tool
def set_reminder(task_description: str, time_str: str) -> str:
    """
    Sets a one-time or recurring reminder.

    For one-time reminders, time_str can be natural language:
      - 'today at 3pm'
      - 'tomorrow at 10am'
      - 'in 2 hours'
      - 'next Monday at 9am'

    For recurring reminders, prefix with 'every':
      - 'every day at 8am'
      - 'every Monday at 9am'
      - 'every weekday at 7:30am'
      - 'every Sunday at 6pm'
    """
    #logger.info(f"Tool called: set_reminder | task='{task_description}' | time='{time_str}'")
    try:
        time_lower = time_str.lower().strip()
        is_recurring = time_lower.startswith("every")

        if is_recurring:
            # --- RECURRING REMINDER ---
            job = _schedule_recurring(task_description, time_lower)
            label = _describe_recurring(time_lower)
            _tasks.append({
                "task": task_description,
                "time_str": time_str,
                "run_time": label,
                "job_id": job.id,
                "recurring": True,
            })
            logger.info(f"Recurring reminder scheduled: {label}")
            return f"Confirmed: I'll remind you to '{task_description}' {label} ({USER_TIMEZONE})."

        else:
            # --- ONE-TIME REMINDER ---
            run_time = parse_local_time(time_str)

            if run_time is None:
                raise ValueError(f"Could not understand the time: '{time_str}'")

            if run_time < now_local():
                raise ValueError(
                    f"'{run_time.strftime('%A, %b %d at %I:%M %p %Z')}' is in the past. Please provide a future time."
                )

            job = _scheduler.add_job(
                _reminder_fired,
                trigger="date",
                run_date=run_time,
                args=[task_description],
                id=f"reminder_{len(_tasks)}",
                timezone=LOCAL_TZ,
            )

            _tasks.append({
                "task": task_description,
                "time_str": time_str,
                "run_time": run_time.strftime("%Y-%m-%d %H:%M %Z"),
                "job_id": job.id,
                "recurring": False,
            })

            logger.info(f"One-time reminder scheduled for {run_time}")
            return f"Confirmed: I'll remind you to '{task_description}' on {run_time.strftime('%A, %b %d at %I:%M %p')} ({USER_TIMEZONE})."

    except ValueError as e:
        logger.warning(f"set_reminder failed: {e}")
        return f"Sorry, I couldn't set that reminder: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in set_reminder: {e}", exc_info=True)
        return "Sorry, an unexpected error occurred while setting your reminder."


def _schedule_recurring(task_description: str, time_lower: str):
    """Parses 'every X at Y' patterns and creates a CronTrigger job."""

    DAY_MAP = {
        "monday": "mon", "tuesday": "tue", "wednesday": "wed",
        "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun",
        "weekday": "mon-fri", "weekend": "sat,sun", "day": "*",
    }

    # Extract hour/minute from time string
    # Parse just the time portion e.g. "8am", "7:30am", "9pm"
    time_part = time_lower.replace("every", "").strip()
    for day_key in DAY_MAP:
        time_part = time_part.replace(day_key, "").strip()
    time_part = time_part.replace("at", "").strip()

    parsed_time = dateparser.parse(
        time_part or "now",
        settings={"RETURN_AS_TIMEZONE_AWARE": False},
    )
    hour = parsed_time.hour if parsed_time else 9
    minute = parsed_time.minute if parsed_time else 0

    # Determine day of week
    day_of_week = "*"
    for key, cron_val in DAY_MAP.items():
        if key in time_lower:
            day_of_week = cron_val
            break

    trigger = CronTrigger(
        day_of_week=day_of_week,
        hour=hour,
        minute=minute,
        timezone=LOCAL_TZ,
    )

    job = _scheduler.add_job(
        _reminder_fired,
        trigger=trigger,
        args=[task_description],
        id=f"reminder_{len(_tasks)}",
    )
    return job


def _describe_recurring(time_lower: str) -> str:
    """Returns a human-readable description of the recurring schedule."""
    if "weekday" in time_lower:
        day_label = "every weekday (Mon–Fri)"
    elif "weekend" in time_lower:
        day_label = "every weekend (Sat–Sun)"
    elif "day" in time_lower and "monday" not in time_lower and "wednesday" not in time_lower:
        day_label = "every day"
    else:
        for day in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]:
            if day in time_lower:
                day_label = f"every {day.capitalize()}"
                break
        else:
            day_label = "on a recurring schedule"
    return day_label


@tool
def get_upcoming_tasks() -> str:
    """Retrieves all scheduled reminders, showing local time and whether they are recurring."""
    #logger.info("Tool called: get_upcoming_tasks")
    try:
        if not _tasks:
            return "Your schedule is currently clear."
        lines = []
        for t in _tasks:
            label = "🔁 Recurring" if t.get("recurring") else "🔔 One-time"
            lines.append(f"{label} | {t['task']} → {t['run_time']}")
        return f"Scheduled reminders ({USER_TIMEZONE}):\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"Unexpected error in get_upcoming_tasks: {e}", exc_info=True)
        return "Sorry, I couldn't retrieve your tasks right now."


@tool
def open_and_search(query: str) -> str:
    """Opens a browser, searches DuckDuckGo, and returns the top result summary."""
    #logger.info(f"Tool called: open_and_search | query='{query}'")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            try:
                page = browser.new_page()
                page.goto(f"https://duckduckgo.com/?q={query}&ia=web", timeout=15000)
                page.wait_for_selector("[data-testid='result']", timeout=10000)
                result = page.inner_text("[data-testid='result']")[:500]
                logger.info("Browser search completed successfully.")
                return f"Search result for '{query}': {result}..."
            except PlaywrightTimeout:
                logger.warning(f"Browser search timed out for query: '{query}'")
                return "Sorry, the search timed out. Please try again."
            except Exception as e:
                logger.error(f"Browser error during search: {e}", exc_info=True)
                return "Sorry, an error occurred during the browser search."
            finally:
                browser.close()
    except Exception as e:
        logger.error(f"Playwright failed to launch: {e}", exc_info=True)
        return "Sorry, the browser could not be launched."


# ---------------------------------------------------------------------------
# 5. LANGGRAPH AGENT
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]


tools = [set_reminder, get_upcoming_tasks, open_and_search]
tool_node = ToolNode(tools)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0).bind_tools(tools)


def call_model(state: AgentState):
    #logger.info("Agent: Invoking LLM")
    try:
        messages = [
            SystemMessage(
                content=system_prompt(
                    current_time=now_local(),
                    timezone=USER_TIMEZONE,
                )
            )
        ] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}", exc_info=True)
        raise

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()


# ---------------------------------------------------------------------------
# 6. EXECUTION INTERFACE
# ---------------------------------------------------------------------------

def run_assistant(user_input: str):
    #logger.info(f"User input: {user_input}")
    print(f"\nUser: {user_input}")
    try:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "agent" and value["messages"][-1].content:
                    response = value["messages"][-1].content
                    #logger.info(f"Assistant response: {response}")
                    print(f"Assistant: {response}")
    except Exception as e:
        logger.error(f"run_assistant failed: {e}", exc_info=True)
        print("Assistant: Sorry, something went wrong. Please try again.")


# ---------------------------------------------------------------------------
# 7. LIVE TEST SCENARIOS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_assistant("Remind me to cook rice around 10am tomorrow")
    #run_assistant("Remind me to drink water every day at 8am")
    #run_assistant("Remind me to call mum every Sunday at 6pm")
    run_assistant("What did i schedule something for tomorrow?") 