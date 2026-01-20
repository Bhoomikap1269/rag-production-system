import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

index = faiss.read_index("vector_store/faiss.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

times = []

for _ in range(100):
    start = time.time()
    q_emb = model.encode(["wireless earbuds"], normalize_embeddings=True)
    index.search(q_emb.astype("float32"), 5)
    times.append((time.time() - start) * 1000)

times = np.array(times)

print("p50 latency (ms):", np.percentile(times, 50))
print("p95 latency (ms):", np.percentile(times, 95))
