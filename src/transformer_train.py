import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
from transformer_reasoner import TaskTransformer
from transformer_features import task_to_vector


DB = "rl_memory.db"


def load_rl_data():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        SELECT text, reward
        FROM rl_events
    """)

    data = c.fetchall()
    conn.close()

    return data


def train_transformer():
    data = load_rl_data()

    if not data:
        print("No RL data found.")
        return

    model = TaskTransformer(input_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    epochs = 20

    for epoch in range(epochs):

        total_loss = 0

        for text, reward in data:

            # fake reconstruction of task vector
            # (we keep it simple for Layer 2.5)
            x = torch.tensor([
                len(str(text)), 0, 0, 0
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            y = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)

            pred = model(x)

            pred = model(x).mean(dim=1)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "transformer_brain.pth")

    print("Transformer trained ✔ saved as transformer_brain.pth")

if __name__ == "__main__":
    train_transformer()