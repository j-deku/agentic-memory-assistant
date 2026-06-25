from tools.task_tools import TaskTools


class ToolExecutor:

    def __init__(self, task_tools:TaskTools):
        self.tools = task_tools

    def run(self, plan):

        results = []

        for step in plan.steps:

            if step.action == "add_task":

                results.append(
                    self.tools.add_task(
                        step.slots.get("title")
                    )
                )

            elif step.action == "complete_task":

                results.append(
                    self.tools.complete_task(
                        step.slots.get("task_id")
                    )
                )

            elif step.action == "delete_task":

                results.append(
                    self.tools.delete_task(
                        step.slots.get("task_id")
                    )
                )

            elif step.action == "list_tasks":

                results.append(
                    self.tools.list_tasks()
                )

        return results