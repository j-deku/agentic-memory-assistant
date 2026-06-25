import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from nlp.torch_dataset import X_tensor, y_tensor, sample_weights
from nlp.intent_model import IntentNet

# =========================
# MODEL SETUP
# =========================
input_dim = X_tensor.shape[1]
num_classes = len(torch.unique(y_tensor))

model = IntentNet(input_dim, num_classes)

# =========================
# LOSS (RL WEIGHTED)
# =========================
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

# convert weights
if "sample_weights" in globals():
    weights = sample_weights
else:
    weights = torch.ones(len(y_tensor))

# =========================
# TRAIN LOOP
# =========================
for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X_tensor)

    loss_raw = criterion(outputs, y_tensor)

    # 🔥 RL weighted loss
    loss = (loss_raw * weights).mean()

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "intent_torch_model.pth")

joblib.dump(weights, "rl_weights.pkl")

print("PyTorch Intent Brain trained ✔ (RL-enabled)")