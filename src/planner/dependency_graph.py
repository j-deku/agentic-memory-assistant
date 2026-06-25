# planner/dependency_graph.py

def build_dependencies(subtasks):
    graph = []

    for i, task in enumerate(subtasks):
        dependencies = []

        if i > 0:
            dependencies.append(subtasks[i - 1])

        graph.append({
            "task": task,
            "depends_on": dependencies
        })

    return graph