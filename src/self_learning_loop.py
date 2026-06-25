from ml_dataset import load_training_data
from train_all import train_all_models

def self_learning_cycle():

    df = load_training_data()

    if len(df) >= 10:
        train_all_models()
        return "Brain updated."

    return "Collecting more experience."