from embeddings.text_embedder import embed_query
from vector_store.faiss_index import faiss_search
from vector_store.bm25_search import bm25_search
from data.processed.doc_store import get_text

def rrf_fusion(dense_results, sparse_results, k=60):
    scores = {}

    for rank, doc in enumerate(dense_results):
        scores.setdefault(doc["id"], 0)
        scores[doc["id"]] += 1 / (k + rank)

    for rank, doc in enumerate(sparse_results):
        scores.setdefault(doc["id"], 0)
        scores[doc["id"]] += 1 / (k + rank)

    return scores

def hybrid_search(query, k=10):
    # Dense retrieval
    query_emb = embed_query(query)
    dense = faiss_search(query_emb, k=20)

    # Sparse retrieval
    sparse = bm25_search(query, k=20)

    fused_scores = rrf_fusion(dense, sparse)

    results = []
    for doc_id, score in fused_scores.items():
        results.append({
            "id": doc_id,
            "text": get_text(doc_id),   # ✅ GUARANTEED STRING
            "hybrid_score": score
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:k]
