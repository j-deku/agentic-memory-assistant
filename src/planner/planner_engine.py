# planner/planner_engine.py

from planner.goal_detector import detect_goal
from planner.task_decomposer import decompose_task
from planner.dependency_graph import build_dependencies


def generate_plan(tasks):
    plan = []

    for task in tasks:
        goal = detect_goal(task)
        steps = decompose_task(task)
        graph = build_dependencies(steps)

        plan.append({
            "task": task[1],
            "goal": goal,
            "steps": graph
        })

    return plan