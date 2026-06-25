from rl_memory import update_rl_memory


def update_rl(text, predicted, actual, confidence):
    update_rl_memory(text, predicted, actual, confidence)