import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VectorMemory:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta: List[Dict] = []

    def add(self, text: str, metadata: dict):
        emb = self.model.encode([text])[0].astype("float32")
        self.index.add(np.array([emb]))
        self.meta.append({
                "text": text,
                "metadata": metadata
            })

    def search(self, query: str, k: int = 5):
        if len(self.meta) == 0:
            return []

        emb = self.model.encode([query])[0].astype("float32")
        D, I = self.index.search(np.array([emb]), k)

        results = []
        for i in I[0]:
            if 0 <= i < len(self.meta):
                results.append(self.meta[i])
        return results