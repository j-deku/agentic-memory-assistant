from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DialogueState:
    intent: Optional[str] = None
    slots: dict = field(default_factory=dict)
    awaiting_slot: Optional[str] = None
    pending_slots: dict = field(default_factory=dict)  
    last_task: Optional[str] = None
    last_task_id: Optional[int] = None

    history: list = field(default_factory=list)
    max_history: int = 8

    last_action: str = ""
    last_action_time: Optional[datetime] = None

    def add_turn(self, role: str, text: str):
        self.history.append({"role": role, "text": text})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def reset(self):
        self.intent = None
        self.slots = {}
        self.awaiting_slot = None
        self.pending_slots = {}