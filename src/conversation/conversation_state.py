class ConversationState:

    def __init__(self):

        self.current_goal = None

        self.pending_confirmation = None

        self.last_task_id = None

        self.last_task_name = None

        self.last_topic = None

        self.last_response = None

        self.history = []

    def remember(self, role, text):

        self.history.append({
            "role": role,
            "text": text
        })

        self.history = self.history[-20:]