import torch
import torch.nn as nn
import torch.optim as optim

from transformer_dataset import build_sequences
from transformer_brain import TaskTransformer

X, y = build_sequences()

model = TaskTransformer(input_dim=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "transformer_brain.pth")

print("Transformer Reasoning Agent trained ✔")