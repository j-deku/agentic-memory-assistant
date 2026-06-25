# response_generator.py

import random


ACKS = [
    "Got it.",
    "Absolutely.",
    "Done.",
    "Sure thing.",
    "I've taken care of that."
]


def generate_response(content):

    prefix = random.choice(ACKS)

    return f"{prefix}\n\n{content}"