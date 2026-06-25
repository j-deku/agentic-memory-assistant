import torch
import numpy as np
from transformer_reasoner import TaskTransformer
from transformer_features import task_to_vector

MODEL_PATH = "transformer_brain.pth"

model = TaskTransformer(input_dim=4)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def load_model():
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        print("⚠ Transformer not trained yet - using random weights")


def predict_transformer_score(task):
    """
    Returns transformer-based priority score (0–1 normalized)
    """

    title = task[1]
    category = task[2]
    due_date = task[3]

    # simple feature engineering (same as training style)
    features = np.array([
        len(str(title)),
        1 if category == "work" else 0,
        1 if category == "health" else 0,
        0  # placeholder (future signal)
    ], dtype=np.float32)

    x = torch.tensor(features).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        score = model(x).mean().item()

    # normalize to 0–1
    return max(0.0, min(1.0, score))
    

def predict_task_transformer(task):
    x = task_to_vector(task).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    # normalize to 0–1
    prob = 1 / (1 + abs(score))

    return prob