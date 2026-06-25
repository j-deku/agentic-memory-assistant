# conversation_memory.py

conversation_state = {
    "pending_action": None,
    "data": {}
}


def set_pending_action(action):
    conversation_state["pending_action"] = action


def get_pending_action():
    return conversation_state["pending_action"]


def clear_pending_action():
    conversation_state["pending_action"] = None
    conversation_state["data"] = {}


def set_memory(key, value):
    conversation_state["data"][key] = value


def get_memory(key):
    return conversation_state["data"].get(key)