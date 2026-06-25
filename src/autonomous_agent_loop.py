import time
from tasks import list_tasks
from ai_brain import hybrid_decision
from planner.planner_engine import generate_plan
from transformer_auto_trainer import retrain_transformer_if_needed


def autonomous_cycle():

    print("\n🤖 AUTONOMOUS AGENT STARTED...\n")

    # =========================
    # CYCLE MEMORY (IMPORTANT FIX)
    # =========================
    executed_tasks = set()

    while True:

        # =========================
        # 1. OBSERVE STATE
        # =========================
        tasks = list_tasks()

        if not tasks:
            print("No tasks found. Sleeping...")
            time.sleep(5)
            continue

        # =========================
        # 2. PLAN (Layer 4)
        # =========================
        plan = generate_plan(tasks)

        # =========================
        # 3. DECIDE PRIORITIES (Layer 3 + ML)
        # =========================
        ranked = []

        for task in tasks:
            decision = hybrid_decision(task)
            ranked.append(decision)

        ranked = sorted(
            ranked,
            key=lambda x: x["final_score"],
            reverse=True
        )

        # =========================
        # 4. EXECUTION SELECTION (FIXED LOGIC)
        # =========================
        top_task = None

        for item in ranked:

            # skip already executed in this session
            if item["task"] in executed_tasks:
                continue

            top_task = item
            executed_tasks.add(item["task"])
            break

        # if everything was executed already → reset cycle memory
        if top_task is None:
            print("\n🔄 All tasks processed in this cycle. Resetting memory...\n")
            executed_tasks.clear()
            time.sleep(2)
            continue

        # =========================
        # 5. EXECUTE (SIMULATED ACTION)
        # =========================
        print("\n🔥 EXECUTING TOP TASK:")
        print("Task:", top_task["task"])
        print("Score:", top_task["final_score"])
        print("Decision:", top_task["decision"])

        # =========================
        # 6. LEARNING CYCLE
        # =========================
        retrain_transformer_if_needed()

        # =========================
        # 7. WAIT (CONTROL LOOP SPEED)
        # =========================
        time.sleep(10)