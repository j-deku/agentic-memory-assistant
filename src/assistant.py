import logging
import sys
from datetime import datetime
from typing import Annotated, TypedDict, List

import dateparser
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------------------------------------------------------------------------
# 1. LOGGING — writes to console AND a file
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("assistant.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("assistant")


# ---------------------------------------------------------------------------
# 2. SCHEDULER — actually fires reminders at the right time
# ---------------------------------------------------------------------------

def _reminder_fired(task_description: str):
    """Callback that runs when a scheduled reminder triggers."""
    logger.info(f"🔔 REMINDER: {task_description}")
    print(f"\n⏰ REMINDER: {task_description}\n")


def _scheduler_listener(event):
    """Listens for scheduler job success/failure and logs accordingly."""
    if event.exception:
        logger.error(f"Scheduled job failed: {event.job_id} — {event.exception}")
    else:
        logger.info(f"Scheduled job completed successfully: {event.job_id}")


_scheduler = BackgroundScheduler()
_scheduler.add_listener(_scheduler_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
_scheduler.start()
_tasks: list = []


# ---------------------------------------------------------------------------
# 3. TOOLS
# ---------------------------------------------------------------------------

@tool
def set_reminder(task_description: str, time_str: str) -> str:
    """Sets a timed reminder. time_str can be natural language like 'tomorrow 10am' or ISO format."""
    logger.info(f"Tool called: set_reminder | task='{task_description}' | time='{time_str}'")
    try:
        # Parse natural language time into a datetime object
        run_time = dateparser.parse(
            time_str,
            settings={
                "PREFER_DATES_FROM": "current_period",
                "RETURN_AS_TIMEZONE_AWARE": False,
                "RELATIVE_BASE": datetime.now(),
            },
        )
        if run_time is None:
            raise ValueError(f"Could not understand the time: '{time_str}'")

        if run_time < datetime.now():
            raise ValueError(f"Parsed time '{run_time}' is in the past. Please provide a future time.")

        # Schedule the actual reminder
        job = _scheduler.add_job(
            _reminder_fired,
            trigger="date",
            run_date=run_time,
            args=[task_description],
            id=f"reminder_{len(_tasks)}",
        )

        _tasks.append({
            "task": task_description,
            "time_str": time_str,
            "run_time": run_time.strftime("%Y-%m-%d %H:%M"),
            "job_id": job.id,
        })

        logger.info(f"Reminder scheduled successfully for {run_time}")
        return f"Confirmed: I'll remind you to '{task_description}' on {run_time.strftime('%A, %b %d at %I:%M %p')}."

    except ValueError as e:
        logger.warning(f"set_reminder failed: {e}")
        return f"Sorry, I couldn't set that reminder: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in set_reminder: {e}", exc_info=True)
        return "Sorry, an unexpected error occurred while setting your reminder."


@tool
def get_upcoming_tasks() -> str:
    """Retrieves all upcoming scheduled reminders."""
    logger.info("Tool called: get_upcoming_tasks")
    try:
        if not _tasks:
            return "Your schedule is currently clear."
        lines = [f"- {t['task']} → {t['run_time']}" for t in _tasks]
        return "Upcoming reminders:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"Unexpected error in get_upcoming_tasks: {e}", exc_info=True)
        return "Sorry, I couldn't retrieve your tasks right now."


@tool
def open_and_search(query: str) -> str:
    """Opens a browser, searches DuckDuckGo, and returns the top result summary."""
    logger.info(f"Tool called: open_and_search | query='{query}'")
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
                return f"Sorry, the search timed out. DuckDuckGo may be slow. Please try again."
            except Exception as e:
                logger.error(f"Browser error during search: {e}", exc_info=True)
                return "Sorry, an error occurred during the browser search."
            finally:
                browser.close()  # Always closes browser even if an error occurs
    except Exception as e:
        logger.error(f"Playwright failed to launch: {e}", exc_info=True)
        return "Sorry, the browser could not be launched."


# ---------------------------------------------------------------------------
# 4. LANGGRAPH AGENT
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]


tools = [set_reminder, get_upcoming_tasks, open_and_search]
tool_node = ToolNode(tools)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0).bind_tools(tools)


def call_model(state: AgentState):
    logger.info("Agent: invoking LLM")
    try:
        response = llm.invoke(state["messages"])
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
# 5. EXECUTION INTERFACE
# ---------------------------------------------------------------------------

def run_assistant(user_input: str):
    logger.info(f"User input: {user_input}")
    print(f"\nUser: {user_input}")
    try:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "agent" and value["messages"][-1].content:
                    response = value["messages"][-1].content
                    logger.info(f"Assistant response: {response}")
                    print(f"Assistant: {response}")
    except Exception as e:
        logger.error(f"run_assistant failed: {e}", exc_info=True)
        print("Assistant: Sorry, something went wrong. Please try again.")


# ---------------------------------------------------------------------------
# 6. LIVE TEST SCENARIOS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    #run_assistant("Remind me to cook food today around 5am")
   run_assistant("What do I have scheduled for today?")
   #run_assistant("Open a browser and search for the definition of breakfast")