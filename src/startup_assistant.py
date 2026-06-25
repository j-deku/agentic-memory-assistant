# startup_assistant.py

from datetime import datetime
from adaptive_personality import get_personalized_context
from assistant_personality import wake_message
from tasks import list_tasks


def startup_message():

    hour = datetime.now().hour

    if hour < 12:
        greeting = "Good morning"
    elif hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    tasks = list_tasks()

    pending = sum(1 for t in tasks if not t[4])

    print("\n🧠 Aerial AI Online \n")
    print(wake_message())
    print(f"\n{greeting}, Jeremiah.")
    print(f"\nYou currently have {pending} pending tasks.")
    context = get_personalized_context()

    if context:
        print(context)
    else:
        print("How can I help you today?")