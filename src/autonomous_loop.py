import time
from datetime import datetime
from agent import autonomous_plan, autonomous_actions
from tasks import update_memory_from_tasks


def run_morning_cycle():
    print("\n🌅 MORNING CYCLE - DAILY PLAN\n")

    plan = autonomous_plan()

    for item in plan:
        print("•", item)

    print("\n⚙️ ACTIONS:\n")
    actions = autonomous_actions()

    for a in actions:
        print("•", a)


def run_evening_cycle():
    print("\n🌙 EVENING CYCLE - REFLECTION\n")

    update_memory_from_tasks()

    print("Memory updated ✔")
    print("System learning from today's behavior...")


def autonomous_loop():
    """
    SAFE AUTONOMOUS LOOP (controlled execution)
    """

    while True:
        hour = datetime.now().hour

        # 🌅 Morning cycle (6AM–11AM)
        if hour >= 6 and hour < 11:
            run_morning_cycle()
            time.sleep(60 * 60 * 6)  # wait 6 hours

        # 🌙 Evening cycle (6PM–10PM)
        elif hour >= 16 and hour < 22:
            run_evening_cycle()
            time.sleep(60 * 60 * 6)

        else:
            print("\n🕒 Idle mode - waiting for active cycle...")
            time.sleep(60 * 30)  # check every 30 mins