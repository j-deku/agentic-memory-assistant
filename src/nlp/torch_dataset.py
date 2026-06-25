import numpy as np
import joblib
import torch
from nlp.embedding_model import encode

data = joblib.load("intent_memory.pkl")

texts = data["texts"]
labels = data["labels"]

# Encode texts into embeddings
X = np.array(encode(texts))

# Convert labels → numeric IDs
label_map = {label: i for i, label in enumerate(sorted(set(labels)))}
y = np.array([label_map[l] for l in labels])

# Save mapping
joblib.dump(label_map, "label_map.pkl")

# Torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print("Dataset ready ✔")