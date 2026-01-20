from rag.retriever import search as semantic_search
from rag.hybrid import hybrid_search
from ranking.reranker import rerank
from rag.generator import generate_answer

def run_pipeline(query, k=5, mode="semantic", generate=True):
    if mode == "hybrid":
        candidates = hybrid_search(query, k * 3)
    else:
        candidates = semantic_search(query, k * 3)

    ranked = rerank(query, candidates)[:k]

    if not generate:
        return ranked

    contexts = [r["text"] for r in ranked]
    answer = generate_answer(query, contexts)

    return {
        "answer": answer,
        "results": ranked
    }
