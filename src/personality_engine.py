import random

PERSONALITY = {
    "name": "Aerial AI",
    "style": "professional"
}


def greeting(name="User"):

    responses = [
        f"Good to see you, {name}.",
        f"Welcome back, {name}.",
        f"I'm ready whenever you are, {name}.",
        f"Hello {name}, how can I help today?"
    ]

    return random.choice(responses)


def task_added_response(task):

    responses = [
        f"Got it. I've added '{task}'.",
        f"Task recorded: {task}.",
        f"'{task}' has been added successfully.",
        f"I'll keep track of '{task}' for you."
    ]

    return random.choice(responses)


def task_completed_response(task):

    responses = [
        f"Nice work completing '{task}'.",
        f"'{task}' marked as complete.",
        f"Progress made. '{task}' is now finished.",
        f"Excellent. '{task}' has been completed."
    ]

    return random.choice(responses)


def no_tasks_response():

    responses = [
        "You're all caught up.",
        "No pending tasks right now.",
        "Everything looks complete.",
        "Nothing urgent is waiting for you."
    ]

    return random.choice(responses)