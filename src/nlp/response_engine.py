import random

class ResponseEngine:
    def __init__(self, user_name: str = ""):
        self.user_name = user_name

    def task_added(self, title, due, category):
        return random.choice([
            f"Done! Added '{title}'.",
            f"Got it — '{title}' is now saved.",
            f"Added '{title}' successfully."
        ])

    def task_list(self, tasks):
        if not tasks:
            return f"No tasks found. Want me to remind you of something?"
        
        lines = [f"You have {len(tasks)} pending tasks:"]

        for t in tasks[:5]:
            lines.append(f"• {t[1]}")

        return "\n".join(lines)

    def task_completed(self, title):
        return f"Nice!. '{title}' has been marked completed ✓"

    def task_deleted(self, title):
        return f"Done. '{title}' has been removed from your tasks list ✓"

    def recommended_tasks(self, tasks):

        if not tasks:
            return "I don't have any recommendations right now."

        lines = ["I recommend these tasks next:"]

        for t in tasks[:5]:
            lines.append(f"• {t[1]}")

        return "\n".join(lines)
    
    def overdue_tasks(self, tasks):

        if not tasks:
            return f"You have no overdue tasks, {self.user_name}."

        return f"You have {len(tasks)} overdue tasks." 
                              
    def check_task(self, title):      
        if not title:
            return f"No, you don't have '{title}' in your task list. "
        
        return f"Yes, {self.user_name}. You have '{title}' in your list. Have you completed that already?"                                                                                                                                