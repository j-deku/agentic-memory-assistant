"""
create_training_data.py  —  Step 2 (manual edition)

Generates 2000 training examples locally with zero API calls.
Covers all edge cases for 3 tools: set_reminder, get_upcoming_reminders, open_and_search.

Usage:
  python create_training_data.py
  Output: training_data.jsonl
"""

import json
import random
from pathlib import Path

OUTPUT_FILE = "training_data.jsonl"
random.seed(42)

# ---------------------------------------------------------------------------
# RAW EXAMPLE POOLS
# Each entry: (user_input, tool_name, args_dict)
# ---------------------------------------------------------------------------

SET_REMINDER_ONE_TIME = [
    # Explicit time phrases
    ("Remind me to call mum at 5pm today",                   "set_reminder", {"task_description": "call mum",               "time_str": "today at 5pm"}),
    ("Set a reminder for my dentist appointment at 9am tomorrow", "set_reminder", {"task_description": "dentist appointment",     "time_str": "tomorrow at 9am"}),
    ("Don't let me forget to pay rent on the 1st at 8am",    "set_reminder", {"task_description": "pay rent",               "time_str": "on the 1st at 8am"}),
    ("Remind me to take my medicine in 30 minutes",          "set_reminder", {"task_description": "take medicine",           "time_str": "in 30 minutes"}),
    ("Alert me to check my email at 2pm",                    "set_reminder", {"task_description": "check email",             "time_str": "at 2pm"}),
    ("Remind me about the team meeting next Monday at 10am", "set_reminder", {"task_description": "team meeting",            "time_str": "next Monday at 10am"}),
    ("Ping me to submit the report by Friday at 3pm",        "set_reminder", {"task_description": "submit the report",       "time_str": "Friday at 3pm"}),
    ("Remind me to charge my laptop tonight at 10pm",        "set_reminder", {"task_description": "charge laptop",           "time_str": "tonight at 10pm"}),
    ("Set an alarm to wake up at 6:30am tomorrow",           "set_reminder", {"task_description": "wake up",                 "time_str": "tomorrow at 6:30am"}),
    ("Remind me to call the bank around lunchtime tomorrow", "set_reminder", {"task_description": "call the bank",           "time_str": "tomorrow around lunchtime"}),
    ("Don't let me forget the gym session at 6am",           "set_reminder", {"task_description": "gym session",             "time_str": "at 6am"}),
    ("Heads up for my flight check-in at 4am Saturday",      "set_reminder", {"task_description": "flight check-in",         "time_str": "Saturday at 4am"}),
    ("Remind me to pick up the kids at 3:30pm",              "set_reminder", {"task_description": "pick up the kids",        "time_str": "at 3:30pm"}),
    ("Set a reminder to review my budget this Sunday at 6pm","set_reminder", {"task_description": "review budget",           "time_str": "this Sunday at 6pm"}),
    ("Remind me about the webinar at 2pm next Tuesday",      "set_reminder", {"task_description": "webinar",                 "time_str": "next Tuesday at 2pm"}),
    ("Can you remind me to send the invoice at 9am Monday",  "set_reminder", {"task_description": "send the invoice",        "time_str": "Monday at 9am"}),
    ("Set a reminder for my anniversary dinner at 7pm Saturday", "set_reminder", {"task_description": "anniversary dinner",  "time_str": "Saturday at 7pm"}),
    ("Remind me to water the plants at 8am tomorrow",        "set_reminder", {"task_description": "water the plants",        "time_str": "tomorrow at 8am"}),
    ("Please remind me to take out the trash tonight at 9pm","set_reminder", {"task_description": "take out the trash",      "time_str": "tonight at 9pm"}),
    ("Remind me to follow up with the client at 11am",       "set_reminder", {"task_description": "follow up with client",   "time_str": "at 11am"}),
    ("Set an alarm for 7am tomorrow morning",                "set_reminder", {"task_description": "wake up alarm",           "time_str": "tomorrow at 7am"}),
    ("Remind me to back up my files at midnight",            "set_reminder", {"task_description": "back up files",           "time_str": "at midnight"}),
    ("Don't let me miss the match at 8pm tonight",           "set_reminder", {"task_description": "watch the match",         "time_str": "tonight at 8pm"}),
    ("Remind me to call the doctor tomorrow at 10am",        "set_reminder", {"task_description": "call the doctor",         "time_str": "tomorrow at 10am"}),
    ("Set a reminder to submit the assignment by 11:59pm tonight", "set_reminder", {"task_description": "submit assignment", "time_str": "tonight at 11:59pm"}),
    ("I need a reminder to cook dinner at 6pm",              "set_reminder", {"task_description": "cook dinner",             "time_str": "at 6pm"}),
    ("Remind me about my therapy session at 3pm Thursday",   "set_reminder", {"task_description": "therapy session",         "time_str": "Thursday at 3pm"}),
    ("Set a reminder to renew my subscription before midnight", "set_reminder", {"task_description": "renew subscription",   "time_str": "before midnight"}),
    ("Please set an alert for my meeting in 2 hours",        "set_reminder", {"task_description": "meeting",                 "time_str": "in 2 hours"}),
    ("Remind me to check the oven in 45 minutes",            "set_reminder", {"task_description": "check the oven",          "time_str": "in 45 minutes"}),
    ("Set a reminder for tomorrow at noon to call the office","set_reminder", {"task_description": "call the office",        "time_str": "tomorrow at noon"}),
    ("Remind me to pick up my prescription at 5pm",          "set_reminder", {"task_description": "pick up prescription",    "time_str": "at 5pm"}),
    ("Alert me at 8pm to take my evening medication",        "set_reminder", {"task_description": "take evening medication", "time_str": "at 8pm"}),
    ("I want a reminder to review the contract at 2pm today","set_reminder", {"task_description": "review the contract",     "time_str": "today at 2pm"}),
    ("Remind me to check my messages in an hour",            "set_reminder", {"task_description": "check messages",          "time_str": "in 1 hour"}),
    ("Set a reminder to leave for the airport at 5am",       "set_reminder", {"task_description": "leave for airport",       "time_str": "at 5am"}),
    ("Remind me to vote tomorrow morning at 7am",            "set_reminder", {"task_description": "vote",                    "time_str": "tomorrow at 7am"}),
    ("Don't let me forget to feed the dog at 7pm",           "set_reminder", {"task_description": "feed the dog",            "time_str": "at 7pm"}),
    ("Set a reminder for the school pickup at 2:45pm",       "set_reminder", {"task_description": "school pickup",           "time_str": "at 2:45pm"}),
    ("Remind me to switch off the stove in 20 minutes",      "set_reminder", {"task_description": "switch off the stove",    "time_str": "in 20 minutes"}),
    ("I have a call at 4pm — remind me 10 minutes before",   "set_reminder", {"task_description": "prepare for 4pm call",   "time_str": "at 3:50pm"}),
    ("Remind me to stretch after waking up tomorrow",        "set_reminder", {"task_description": "stretch",                 "time_str": "tomorrow morning"}),
    ("Set an alert for the product launch at 10am Wednesday","set_reminder", {"task_description": "product launch",          "time_str": "Wednesday at 10am"}),
    ("Remind me to reply to John's email by end of day",     "set_reminder", {"task_description": "reply to John's email",   "time_str": "by end of day"}),
    ("Can you set a reminder to pay my electricity bill Friday?", "set_reminder", {"task_description": "pay electricity bill", "time_str": "Friday"}),
    ("Alert me when it's 6am tomorrow",                      "set_reminder", {"task_description": "wake up",                 "time_str": "tomorrow at 6am"}),
    ("Remind me to take a break in 90 minutes",              "set_reminder", {"task_description": "take a break",            "time_str": "in 90 minutes"}),
    ("Set a reminder to call the plumber at 9am",            "set_reminder", {"task_description": "call the plumber",        "time_str": "at 9am"}),
    ("Remind me to start the presentation prep at 1pm",      "set_reminder", {"task_description": "presentation prep",       "time_str": "at 1pm"}),
    ("I need to remember to log my hours at 5pm today",      "set_reminder", {"task_description": "log hours",               "time_str": "today at 5pm"}),
]

SET_REMINDER_RECURRING = [
    ("Remind me to drink water every day at 8am",            "set_reminder", {"task_description": "drink water",             "time_str": "every day at 8am"}),
    ("Set a daily reminder to meditate at 7am",              "set_reminder", {"task_description": "meditate",                "time_str": "every day at 7am"}),
    ("Remind me to call mum every Sunday at 6pm",            "set_reminder", {"task_description": "call mum",               "time_str": "every Sunday at 6pm"}),
    ("Every weekday at 9am remind me to check my tasks",     "set_reminder", {"task_description": "check tasks",             "time_str": "every weekday at 9am"}),
    ("Set a reminder every Monday at 8am for the standup",   "set_reminder", {"task_description": "weekly standup",          "time_str": "every Monday at 8am"}),
    ("Remind me to take vitamins every morning at 7:30am",   "set_reminder", {"task_description": "take vitamins",           "time_str": "every day at 7:30am"}),
    ("Every Friday at 4pm remind me to send the weekly report", "set_reminder", {"task_description": "send weekly report",  "time_str": "every Friday at 4pm"}),
    ("Set a recurring reminder to walk every day at 6pm",    "set_reminder", {"task_description": "go for a walk",           "time_str": "every day at 6pm"}),
    ("Remind me to journal every night at 10pm",             "set_reminder", {"task_description": "journal",                 "time_str": "every day at 10pm"}),
    ("Every Saturday at 10am remind me to clean the house",  "set_reminder", {"task_description": "clean the house",         "time_str": "every Saturday at 10am"}),
    ("Set a recurring alarm for every weekday at 7am",       "set_reminder", {"task_description": "wake up",                 "time_str": "every weekday at 7am"}),
    ("Remind me to review goals every Sunday at 9am",        "set_reminder", {"task_description": "review goals",            "time_str": "every Sunday at 9am"}),
    ("Ping me every day at noon to eat lunch",               "set_reminder", {"task_description": "eat lunch",               "time_str": "every day at 12pm"}),
    ("Every Tuesday and Thursday at 6am remind me to work out", "set_reminder", {"task_description": "work out",            "time_str": "every Tuesday at 6am"}),
    ("Set a reminder to call dad every weekend at 5pm",      "set_reminder", {"task_description": "call dad",               "time_str": "every weekend at 5pm"}),
    ("Remind me to floss every night at 9pm",                "set_reminder", {"task_description": "floss",                   "time_str": "every day at 9pm"}),
    ("Set a daily 6am alarm to wake up",                     "set_reminder", {"task_description": "wake up",                 "time_str": "every day at 6am"}),
    ("Remind me to post on Instagram every day at 7pm",      "set_reminder", {"task_description": "post on Instagram",       "time_str": "every day at 7pm"}),
    ("Set a weekly reminder every Wednesday at 3pm for team sync", "set_reminder", {"task_description": "team sync",        "time_str": "every Wednesday at 3pm"}),
    ("Every morning at 8am remind me to read the news",      "set_reminder", {"task_description": "read the news",           "time_str": "every day at 8am"}),
    ("Remind me to check my budget every Sunday evening",    "set_reminder", {"task_description": "check budget",            "time_str": "every Sunday at 6pm"}),
    ("Set a recurring alert every Monday morning for weekly planning", "set_reminder", {"task_description": "weekly planning", "time_str": "every Monday at 9am"}),
    ("Remind me to stretch every day at 6:30am",             "set_reminder", {"task_description": "stretch",                 "time_str": "every day at 6:30am"}),
    ("I want a daily reminder to pray at 5am",               "set_reminder", {"task_description": "pray",                    "time_str": "every day at 5am"}),
    ("Set a recurring reminder every Friday at 5pm to wrap up work", "set_reminder", {"task_description": "wrap up work",   "time_str": "every Friday at 5pm"}),
    ("Remind me to take a walk every lunchtime at 1pm",      "set_reminder", {"task_description": "take a walk",             "time_str": "every day at 1pm"}),
    ("Every weekday evening at 7pm remind me to cook dinner","set_reminder", {"task_description": "cook dinner",             "time_str": "every weekday at 7pm"}),
    ("Set a reminder to water plants every Monday and Thursday at 8am", "set_reminder", {"task_description": "water plants", "time_str": "every Monday at 8am"}),
    ("Remind me to do pushups every morning at 7am",         "set_reminder", {"task_description": "do pushups",              "time_str": "every day at 7am"}),
    ("Set a nightly reminder to review tomorrow's tasks at 9pm", "set_reminder", {"task_description": "review tomorrow's tasks", "time_str": "every day at 9pm"}),
    ("Remind me every weekday at 8:45am to commute",         "set_reminder", {"task_description": "commute",                 "time_str": "every weekday at 8:45am"}),
    ("I need a recurring reminder to back up my work every Friday at 4pm", "set_reminder", {"task_description": "back up work", "time_str": "every Friday at 4pm"}),
    ("Set a Sunday 8pm reminder to prep for the week",       "set_reminder", {"task_description": "prep for the week",       "time_str": "every Sunday at 8pm"}),
    ("Remind me to check my investments every Saturday morning", "set_reminder", {"task_description": "check investments",   "time_str": "every Saturday at 9am"}),
    ("Every day at 10pm remind me to wind down and sleep",   "set_reminder", {"task_description": "wind down and sleep",     "time_str": "every day at 10pm"}),
    ("Set a recurring Wednesday reminder at 2pm for 1-on-1s","set_reminder", {"task_description": "1-on-1 meeting",         "time_str": "every Wednesday at 2pm"}),
    ("Remind me to meal prep every Sunday at 4pm",           "set_reminder", {"task_description": "meal prep",               "time_str": "every Sunday at 4pm"}),
    ("Set a daily reminder at 11pm to take my night meds",   "set_reminder", {"task_description": "take night medication",   "time_str": "every day at 11pm"}),
    ("Every Thursday at 6pm remind me to take out the bins", "set_reminder", {"task_description": "take out the bins",       "time_str": "every Thursday at 6pm"}),
    ("Remind me to call my team lead every Monday at 10am",  "set_reminder", {"task_description": "call team lead",          "time_str": "every Monday at 10am"}),
    ("Set a recurring morning reminder at 6am every weekday","set_reminder", {"task_description": "morning routine",         "time_str": "every weekday at 6am"}),
    ("I want daily reminders to drink green tea at 3pm",     "set_reminder", {"task_description": "drink green tea",         "time_str": "every day at 3pm"}),
    ("Remind me every Friday to submit my timesheet by 4pm", "set_reminder", {"task_description": "submit timesheet",        "time_str": "every Friday at 4pm"}),
    ("Set a recurring reminder to meditate before bed every day", "set_reminder", {"task_description": "meditate",           "time_str": "every day at 10pm"}),
    ("Every Saturday at 7am remind me to go to the market",  "set_reminder", {"task_description": "go to market",           "time_str": "every Saturday at 7am"}),
    ("Remind me to do a weekly review every Sunday at 7pm",  "set_reminder", {"task_description": "weekly review",           "time_str": "every Sunday at 7pm"}),
    ("Set a reminder to call clients every Tuesday at 11am", "set_reminder", {"task_description": "call clients",            "time_str": "every Tuesday at 11am"}),
    ("Remind me daily at 8:30am to check Slack",             "set_reminder", {"task_description": "check Slack",             "time_str": "every day at 8:30am"}),
    ("Every night at 11pm remind me to charge my devices",   "set_reminder", {"task_description": "charge devices",          "time_str": "every day at 11pm"}),
    ("Set a recurring alert every Monday at 7am to plan the week", "set_reminder", {"task_description": "plan the week",    "time_str": "every Monday at 7am"}),
]

GET_UPCOMING_REMINDERS = [
    ("What reminders do I have?",                            "get_upcoming_reminders", {}),
    ("Show me my scheduled reminders",                       "get_upcoming_reminders", {}),
    ("List my upcoming reminders",                           "get_upcoming_reminders", {}),
    ("Do I have any reminders set?",                         "get_upcoming_reminders", {}),
    ("What did I schedule for tomorrow?",                    "get_upcoming_reminders", {}),
    ("Check my reminders",                                   "get_upcoming_reminders", {}),
    ("What's on my reminder list?",                          "get_upcoming_reminders", {}),
    ("Have I set any reminders today?",                      "get_upcoming_reminders", {}),
    ("What alerts have I got scheduled?",                    "get_upcoming_reminders", {}),
    ("What's coming up in my reminders?",                    "get_upcoming_reminders", {}),
    ("Give me a rundown of my reminders",                    "get_upcoming_reminders", {}),
    ("Any reminders I should know about?",                   "get_upcoming_reminders", {}),
    ("What reminders are active right now?",                 "get_upcoming_reminders", {}),
    ("Show me everything I've been reminded to do",          "get_upcoming_reminders", {}),
    ("What have I set reminders for?",                       "get_upcoming_reminders", {}),
    ("Pull up my reminder list",                             "get_upcoming_reminders", {}),
    ("What's scheduled for me?",                             "get_upcoming_reminders", {}),
    ("Am I forgetting anything?",                            "get_upcoming_reminders", {}),
    ("What do I need to be reminded about?",                 "get_upcoming_reminders", {}),
    ("Show my alerts",                                       "get_upcoming_reminders", {}),
    ("What upcoming reminders have I got?",                  "get_upcoming_reminders", {}),
    ("Can you show me my reminders?",                        "get_upcoming_reminders", {}),
    ("List all my active reminders",                         "get_upcoming_reminders", {}),
    ("What's on my schedule?",                               "get_upcoming_reminders", {}),
    ("Do I have anything set for this week?",                "get_upcoming_reminders", {}),
    ("Show me what reminders I've set so far",               "get_upcoming_reminders", {}),
    ("What notifications am I expecting?",                   "get_upcoming_reminders", {}),
    ("Tell me my reminders",                                 "get_upcoming_reminders", {}),
    ("What's been scheduled for me?",                        "get_upcoming_reminders", {}),
    ("Have I got any alerts coming up?",                     "get_upcoming_reminders", {}),
    ("Check what reminders are pending",                     "get_upcoming_reminders", {}),
    ("What reminders are lined up?",                         "get_upcoming_reminders", {}),
    ("Show pending reminders",                               "get_upcoming_reminders", {}),
    ("What tasks have I set reminders for?",                 "get_upcoming_reminders", {}),
    ("Give me my reminder schedule",                         "get_upcoming_reminders", {}),
    ("What do I have set for today?",                        "get_upcoming_reminders", {}),
    ("Is there anything I set a reminder for?",              "get_upcoming_reminders", {}),
    ("What's on my alert list?",                             "get_upcoming_reminders", {}),
    ("Remind me what my reminders are",                      "get_upcoming_reminders", {}),
    ("What alarms have I got?",                              "get_upcoming_reminders", {}),
    ("Show all scheduled reminders",                         "get_upcoming_reminders", {}),
    ("What reminders did I create?",                         "get_upcoming_reminders", {}),
    ("What's coming up for me?",                             "get_upcoming_reminders", {}),
    ("Check if I have any reminders",                        "get_upcoming_reminders", {}),
    ("Give me an overview of my reminders",                  "get_upcoming_reminders", {}),
    ("What have I scheduled so far?",                        "get_upcoming_reminders", {}),
    ("Show me my notification schedule",                     "get_upcoming_reminders", {}),
    ("Any pending reminders?",                               "get_upcoming_reminders", {}),
    ("What do I need to remember?",                          "get_upcoming_reminders", {}),
    ("Pull up all my reminders",                             "get_upcoming_reminders", {}),
]

OPEN_AND_SEARCH = [
    ("Search for the latest news on AI",                     "open_and_search", {"query": "latest news on AI"}),
    ("Look up the weather in Accra",                         "open_and_search", {"query": "weather in Accra"}),
    ("Find me information about Python asyncio",             "open_and_search", {"query": "Python asyncio"}),
    ("Search online for best productivity apps 2025",        "open_and_search", {"query": "best productivity apps 2025"}),
    ("What is the current Bitcoin price?",                   "open_and_search", {"query": "current Bitcoin price"}),
    ("Look up how to make jollof rice",                      "open_and_search", {"query": "how to make jollof rice"}),
    ("Search for flights from Accra to London",              "open_and_search", {"query": "flights from Accra to London"}),
    ("Find the latest Ghana news",                           "open_and_search", {"query": "Ghana news today"}),
    ("Search for tips on how to sleep better",               "open_and_search", {"query": "tips to sleep better"}),
    ("Look up the population of Ghana",                      "open_and_search", {"query": "population of Ghana"}),
    ("Search for machine learning tutorials for beginners",  "open_and_search", {"query": "machine learning tutorials for beginners"}),
    ("Find a good recipe for banana bread",                  "open_and_search", {"query": "banana bread recipe"}),
    ("Look up symptoms of vitamin D deficiency",             "open_and_search", {"query": "symptoms of vitamin D deficiency"}),
    ("Search for the best budget laptops in 2025",           "open_and_search", {"query": "best budget laptops 2025"}),
    ("What is the current time in New York?",                "open_and_search", {"query": "current time in New York"}),
    ("Look up how to install Docker on Windows",             "open_and_search", {"query": "how to install Docker on Windows"}),
    ("Search for Python fine-tuning tutorials",              "open_and_search", {"query": "Python fine-tuning tutorials"}),
    ("Find the exchange rate between GHS and USD",           "open_and_search", {"query": "GHS to USD exchange rate"}),
    ("What are the best practices for REST APIs?",           "open_and_search", {"query": "REST API best practices"}),
    ("Look up the FIFA World Cup 2026 schedule",             "open_and_search", {"query": "FIFA World Cup 2026 schedule"}),
    ("Search for remote jobs in software engineering",       "open_and_search", {"query": "remote software engineering jobs"}),
    ("Find me a good Python book for beginners",             "open_and_search", {"query": "best Python books for beginners"}),
    ("What is the capital of Australia?",                    "open_and_search", {"query": "capital of Australia"}),
    ("Search for how to lose weight fast",                   "open_and_search", {"query": "how to lose weight fast"}),
    ("Look up the best VPN services in 2025",                "open_and_search", {"query": "best VPN services 2025"}),
    ("Find me a recipe for chicken soup",                    "open_and_search", {"query": "chicken soup recipe"}),
    ("Search for the top universities in Africa",            "open_and_search", {"query": "top universities in Africa"}),
    ("What's the score of the latest Premier League match?", "open_and_search", {"query": "latest Premier League scores"}),
    ("Look up how to use LangChain with Python",             "open_and_search", {"query": "LangChain Python tutorial"}),
    ("Search for the best noise cancelling headphones",      "open_and_search", {"query": "best noise cancelling headphones 2025"}),
    ("Find information about the history of Ghana",          "open_and_search", {"query": "history of Ghana"}),
    ("Look up what OWASP Top 10 is",                         "open_and_search", {"query": "OWASP Top 10"}),
    ("Search for how to negotiate a salary",                 "open_and_search", {"query": "how to negotiate salary"}),
    ("What are the side effects of ibuprofen?",              "open_and_search", {"query": "side effects of ibuprofen"}),
    ("Find me the best time to visit Japan",                 "open_and_search", {"query": "best time to visit Japan"}),
    ("Search for how to start a podcast",                    "open_and_search", {"query": "how to start a podcast"}),
    ("Look up the meaning of cognitive dissonance",          "open_and_search", {"query": "cognitive dissonance meaning"}),
    ("Find news about Elon Musk today",                      "open_and_search", {"query": "Elon Musk news today"}),
    ("Search for the best online courses for data science",  "open_and_search", {"query": "best online data science courses"}),
    ("What is the latest version of Python?",                "open_and_search", {"query": "latest Python version"}),
    ("Look up how to build a REST API with FastAPI",         "open_and_search", {"query": "FastAPI REST API tutorial"}),
    ("Search for affordable hotels in Dubai",                "open_and_search", {"query": "affordable hotels in Dubai"}),
    ("Find the definition of machine learning",              "open_and_search", {"query": "machine learning definition"}),
    ("Look up the best ways to save money",                  "open_and_search", {"query": "best ways to save money"}),
    ("Search for how to write a cover letter",               "open_and_search", {"query": "how to write a cover letter"}),
    ("What movies are showing this weekend?",                "open_and_search", {"query": "movies showing this weekend"}),
    ("Look up symptoms of malaria",                          "open_and_search", {"query": "symptoms of malaria"}),
    ("Search for free Python projects for beginners",        "open_and_search", {"query": "Python projects for beginners"}),
    ("Find the latest news about OpenAI",                    "open_and_search", {"query": "OpenAI news"}),
    ("Look up how to do intermittent fasting",               "open_and_search", {"query": "how to do intermittent fasting"}),
]

# ---------------------------------------------------------------------------
# CHATML CONVERSION
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are a helpful assistant with access to these tools: "
    "set_reminder, get_upcoming_reminders, open_and_search. "
    "Respond with a tool call in JSON format: "
    '{"name": "<tool_name>", "arguments": {<args>}}'
)

def to_chatml(user: str, tool: str, args: dict) -> dict:
    tool_call = json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)
    return {
        "conversations": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": tool_call},
        ]
    }

# ---------------------------------------------------------------------------
# BUILD 2000 EXAMPLES via weighted sampling with variation
# ---------------------------------------------------------------------------

def add_variation(user: str) -> str:
    """Randomly rephrase the start of some inputs for variety."""
    prefixes = [
        "", "", "",  # most stay as-is
        "Hey, ", "Please ", "Can you ", "Could you ", "I need you to ",
        "I want to ", "I'd like to ", "Quick question — ",
    ]
    prefix = random.choice(prefixes)
    if prefix and not user[0].isupper():
        return prefix + user
    if prefix:
        return prefix + user[0].lower() + user[1:]
    return user


def main():
    all_pools = [
        (SET_REMINDER_ONE_TIME,  600),   # 30%
        (SET_REMINDER_RECURRING, 600),   # 30%
        (GET_UPCOMING_REMINDERS, 400),   # 20%
        (OPEN_AND_SEARCH,        400),   # 20%
    ]

    examples = []
    for pool, target in all_pools:
        # Sample with replacement to hit the target count
        chosen = random.choices(pool, k=target)
        for user, tool, args in chosen:
            varied_user = add_variation(user)
            examples.append(to_chatml(varied_user, tool, args))

    # Shuffle so tool types are interleaved
    random.shuffle(examples)

    output_path = Path(OUTPUT_FILE)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Done. {len(examples)} examples written to {output_path.resolve()}")
    print(f"\nDistribution:")
    from collections import Counter
    tools = Counter()
    for ex in examples:
        tool_name = json.loads(ex["conversations"][2]["content"])["name"]
        tools[tool_name] += 1
    for tool, count in tools.most_common():
        print(f"  {tool:35s} {count:4d}  ({100*count/len(examples):.1f}%)")
    print(f"\nNext step: run inspect_data.py to verify, then finetune.py on RunPod.")


if __name__ == "__main__":
    main()