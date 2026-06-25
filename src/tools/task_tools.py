class TaskTools:

    def __init__(self, fns: dict):
        self.fns = fns

    def add_task(self, title):
        return self.fns["add_task"](title, "personal", None)

    def delete_task(self, task_id):
        return self.fns["delete_task"](task_id)

    def complete_task(self, task_id):
        return self.fns["complete_task"](task_id)

    def list_tasks(self):
        return self.fns["list_tasks"]()