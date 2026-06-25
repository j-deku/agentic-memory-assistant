from datetime import datetime

def calculate_reward(task):

    reward = 0

    if task["completed"]:
        reward += 10

    if task["days_left"] >= 0:
        reward += 5

    if task["days_left"] < 0:
        reward -= 5

    return reward