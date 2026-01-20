from fastapi import FastAPI
from pydantic import BaseModel
from rag.pipeline import run_pipeline
import time
import json
import os

app = FastAPI(title="RAG++ Search API")

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: str = "semantic"
    generate: bool = True

@app.on_event("startup")
def warmup():
    print("🔥 Warming up RAG++ pipeline...")
    run_pipeline("wireless waterproof bluetooth speaker", k=3)
    print("✅ Warmup complete")


@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    response = run_pipeline(
        req.query,
        k=req.k,
        mode=req.mode,
        generate=req.generate
    )

    latency_ms = (time.time() - start) * 1000

    # Normalize response
    if isinstance(response, dict):
        answer = response["answer"]
        results = response["results"]
    else:
        answer = None
        results = response

    os.makedirs("service", exist_ok=True)
    log = {
        "query": req.query,
        "latency_ms": latency_ms,
        "num_results": len(results),
        "mode": req.mode,
        "generate": req.generate
    }

    with open("service/query_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    return {
        "query": req.query,
        "latency_ms": latency_ms,
        "answer": answer,
        "results": results
    }
