from train_all import train_all_models
from model_monitor import model_needs_retrain


def retrain_if_needed():

    should_train, reason = model_needs_retrain()

    if should_train:
        print(f"\n🧠 Auto-Retraining Triggered: {reason}\n")

        train_all_models()

        return "Model updated ✔"

    return "No retraining needed"