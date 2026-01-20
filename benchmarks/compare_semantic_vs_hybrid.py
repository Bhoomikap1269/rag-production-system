from rag.pipeline import run_pipeline
import random

# Simple evaluation set (reuse corpus queries)
queries = [
    "ipx7 speaker",
    "jbl xtreme",
    "coleman waterproof",
    "bh50 bluetooth",
    "renewed speaker blue"
]


def evaluate(mode):
    hits = 0
    total = 0

    for q in queries:
        results = run_pipeline(q, k=3, mode=mode)
        for r in results:
            if "speaker" in r["text"].lower():
                hits += 1
                break
        total += 1

    return hits / total

if __name__ == "__main__":
    sem = evaluate("semantic")
    hyb = evaluate("hybrid")

    print(f"Semantic Recall@5: {sem:.2f}")
    print(f"Hybrid   Recall@5: {hyb:.2f}")
