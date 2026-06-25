import joblib
from nlp.training_data import training_examples
from nlp.embedding_model import encode

texts = [t[0] for t in training_examples]
labels = [t[1] for t in training_examples]

embeddings = encode(texts)

joblib.dump({
    "embeddings": embeddings,
    "labels": labels,
    "texts": texts
}, "intent_memory.pkl")

print("Embedding brain trained ✔")