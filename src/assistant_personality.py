# assistant_personality.py

import random

WAKE_MESSAGES = [
    "Ready to assist.",
    "All systems operational.",
    "Monitoring your productivity.",
    "Learning from recent activity.",
    "Transformer reasoning active.",
    "Planning engine online."
]


def wake_message():
    return random.choice(WAKE_MESSAGES)