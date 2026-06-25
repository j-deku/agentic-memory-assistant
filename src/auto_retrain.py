import os
import time

THRESHOLD = 20  # events before retrain


def should_retrain(event_count):
    return event_count >= THRESHOLD


def retrain_if_needed():
    from rl_dataset_builder import load_rl_data

    df = load_rl_data()

    if len(df) > THRESHOLD:
        print("🔁 Retraining triggered...")
        os.system("python nlp/train_torch_intent.py")