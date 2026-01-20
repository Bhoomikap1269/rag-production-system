import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_FILE = "data/processed/products.jsonl"
INDEX_FILE = "vector_store/faiss.index"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)

docs = []
with open(DATA_FILE) as f:
    for line in f:
        docs.append(json.loads(line))

def recall_at_k(k=5, samples=200):
    hits = 0

    for i in range(samples):
        query = docs[i]["text"]
        q_emb = model.encode([query], normalize_embeddings=True)

        _, ids = index.search(q_emb.astype("float32"), k)

        if i in ids[0]:
            hits += 1

    return hits / samples

for k in [1, 3, 5, 10]:
    print(f"Recall@{k}: {recall_at_k(k):.3f}")
