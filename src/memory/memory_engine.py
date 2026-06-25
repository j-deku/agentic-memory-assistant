class MemoryEngine:

    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def update_from_tasks(self, tasks):
        self.store["last_task_count"] = len(tasks)