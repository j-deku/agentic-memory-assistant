import numpy as np
import joblib
from nlp.embedding_model import encode

# =========================
# LOAD TRAINED MEMORY
# =========================
data = joblib.load("intent_memory.pkl")

embeddings = np.array(data["embeddings"])
labels = np.array(data["labels"])


# =========================
# SIMILARITY FUNCTION
# (cosine for normalized embeddings = dot product)
# =========================
def similarity(a, b):
    return np.dot(a, b)


# =========================
# SOFT CONFIDENCE SCALING
# =========================
def normalize(score):
    return 1 / (1 + np.exp(-10 * (score - 0.5)))


# =========================
# MAIN PREDICTOR
# =========================
def predict_intent(text):
    query = encode([text])[0]

    # =========================
    # FAST VECTOR SEARCH
    # =========================
    scores = embeddings @ query   # matrix multiplication

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    best_label = labels[best_index]

    # =========================
    # UNKNOWN HANDLING
    # =========================
    if best_score < 0.45:
        return "unknown", best_score

    confidence = normalize(best_score)

    return best_label, confidence