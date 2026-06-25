import torch
import numpy as np
import joblib
from nlp.embedding_model import encode
from nlp.intent_model import IntentNet

label_map = joblib.load("label_map.pkl")
reverse_map = {v: k for k, v in label_map.items()}

input_dim = 384  # MiniLM dimension
num_classes = len(label_map)

model = IntentNet(input_dim, num_classes)
model.load_state_dict(torch.load("intent_torch_model.pth"))
model.eval()


def predict_intent(text):
    with torch.no_grad():

        embedding = np.array(encode([text])[0])
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

        intent = reverse_map[int(pred.item())]

        if confidence.item() < 0.5:
            return "unknown", float(confidence.item())

        return intent, float(confidence.item())