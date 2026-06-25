class TaskService:
    def __init__(self, fns):
        self.fns = fns

    def list_tasks(self):
        return self.fns["list_tasks"]()

    def add(self, title, category, due):
        return self.fns["add_task"](title, category, due)

    def complete(self, task_id):
        return self.fns["complete_task"](task_id)

    def delete(self, task_id):
        return self.fns["delete_task"](task_id)

    def productivity_score(self):
        return self.fns["productivity_score"]()

    def overdue_tasks(self):
        return self.fns["get_overdue_tasks"]()

    def recommended_tasks(self):
        return self.fns["get_recommended_tasks"]()

    def analyze_habits(self):
        return self.fns["analyze_habits"]()