from sentence_transformers import CrossEncoder

# Cross-encoder for high-precision reranking
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, top_k=5):
    """
    candidates: list of dicts with keys {id, score, text}
    """
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
