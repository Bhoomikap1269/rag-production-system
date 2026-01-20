import pickle

INDEX_PATH = "vector_store/bm25.pkl"

with open(INDEX_PATH, "rb") as f:
    bm25, documents = pickle.load(f)

def bm25_search(query, k=10):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    results = []
    for idx, score in ranked:
        doc = documents[idx]
        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "bm25_score": float(score)
        })

    return results

