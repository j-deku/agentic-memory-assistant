from ml_dataset import load_training_data
from classifier import train_classifier
from regressor import train_regressor


def train_all_models():
    df = load_training_data()

    print("Dataset size:", len(df))

    if len(df) < 3:
        print("Not enough data yet")
        return

    train_classifier(df)
    train_regressor(df)

    print("Both models trained ✔")


if __name__ == "__main__":
    train_all_models()