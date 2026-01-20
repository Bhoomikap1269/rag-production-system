import faiss
import numpy as np

EMB_FILE = "data/processed/text_embeddings.npy"
IDS_FILE = "data/processed/text_ids.npy"
INDEX_FILE = "vector_store/faiss.index"

# Load embeddings and ids
embeddings = np.load(EMB_FILE).astype("float32")
ids = np.load(IDS_FILE)

dim = embeddings.shape[1]

# Build index
index = faiss.IndexFlatIP(dim)   # cosine similarity (normalized vectors)
index.add(embeddings)

# Persist index
faiss.write_index(index, INDEX_FILE)

print("FAISS index built")
print("Vectors:", index.ntotal)

# -------------------------
# Runtime retrieval function
# -------------------------

def faiss_search(query_embedding, k=10):
    """
    Search FAISS index using a query embedding.
    """
    scores, indices = index.search(query_embedding, k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        results.append({
            "id": int(ids[i]),
            "score": float(score)
        })

    return results
