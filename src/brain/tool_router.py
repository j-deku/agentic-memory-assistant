from brain.goal_types import Goal


class ToolRouter:

    def __init__(self, task_functions):

        self.task_functions = task_functions

    def execute(self,
                reasoning):

        goal = reasoning["goal"]

        if goal == Goal.VIEW_TASKS:

            return self.task_functions[
                "list_tasks"
            ]()

        return None