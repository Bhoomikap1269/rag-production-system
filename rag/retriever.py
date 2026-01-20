import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "vector_store/faiss.index"
DATA_FILE = "data/processed/products.jsonl"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)

documents = {}
with open(DATA_FILE) as f:
    for line in f:
        obj = json.loads(line)
        documents[obj["id"]] = obj["text"]

def search(query, k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype("float32"), k)

    results = []
    for i, score in zip(ids[0], scores[0]):
        results.append({
            "id": int(i),
            "score": float(score),
            "text": documents[i][:200]
        })
    return results

if __name__ == "__main__":
    q = "bluetooth speaker waterproof"
    res = search(q)
    for r in res:
        print(r)
