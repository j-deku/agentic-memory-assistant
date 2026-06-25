import torch
import numpy as np
from datetime import datetime


def task_to_vector(task):
    """
    Convert task → numeric vector for transformer
    """

    title_len = len(task["title"])
    category = hash(task["category"]) % 10

    due = task["due_date"]
    days_left = (due - datetime.now()).days if hasattr(due, "days") else 0

    completed = task["completed"]

    return torch.tensor([
        title_len,
        category,
        days_left,
        completed
    ], dtype=torch.float32)