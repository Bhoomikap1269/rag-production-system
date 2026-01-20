# RAG++ — Production-Grade Retrieval-Augmented Generation System

RAG++ is a fully production-ready Retrieval-Augmented Generation (RAG) system designed to mirror real-world **search, recommendation, and question-answering infrastructure** used in large-scale AI platforms.

The project emphasizes **retrieval quality, latency, system design clarity, and deployability**, rather than demo-only abstractions.

---

## Key Features

### Hybrid Retrieval
- Semantic search using **FAISS** (Approximate Nearest Neighbors)
- Lexical search using **BM25**
- Fusion strategy combining semantic and keyword signals

### High-Precision Ranking
- Cross-encoder re-ranking for improved relevance
- Optimized candidate selection before generation

### Grounded LLM Answer Generation
- Instruction-tuned **FLAN-T5** for context-aware answers
- Answers generated strictly from retrieved documents

### Production API
- **FastAPI** service with structured request and response schema
- Warmup handling to reduce cold-start latency
- Query logging for observability

### Performance & Evaluation
- Recall@K benchmarking
- End-to-end latency measurement
- Stable sub-second responses on CPU

### Deployment-Ready
- Fully Dockerized
- No external API dependencies
- Reproducible local and container execution

---

## System Architecture

User Query

↓

Semantic Retrieval (FAISS)

Lexical Retrieval (BM25)

↓

Hybrid Fusion

↓

Cross-Encoder Re-Ranking

↓

Context Selection

↓

LLM Answer Generation (FLAN-T5)

↓

FastAPI Service

↓

Docker Deployment

---

## Dataset

- Amazon product listings (Kaggle)
- ~7,000 real-world product descriptions
- Noisy, unstructured text to simulate real search workloads

---


---

## Dataset

- Amazon product listings (Kaggle)
- ~7,000 real-world product descriptions
- Noisy, unstructured text to simulate real search workloads

---

## API Usage

### Search Endpoint

**POST** `/search`

#### Request

```json

{
  "query": "iphone waterproof bluetooth speaker",
  "k": 5,
  "mode": "hybrid",
  "generate": true
}

```

### Response

```json

{
  "query": "iphone waterproof bluetooth speaker",
  "latency_ms": 520.3,
  "answer": "A good waterproof Bluetooth speaker option is the JBL Xtreme 2...",
  "results": [
    {
      "id": 472,
      "text": "JBL Xtreme 2 Portable Waterproof Wireless Bluetooth Speaker - Blue",
      "hybrid_score": 0.029,
      "rerank_score": 4.86
    }
  ]
}

```

---

### Running Locally

python -m uvicorn service.app:app --reload

---

### Running with Docker

docker build -t rag-production-system .
docker run -p 8000:8000 rag-production-system

---

### Why This Project Matters

This project reflects real-world RAG system design by emphasizing:

Retrieval correctness over prompt engineering

Ranking quality before generation

Latency stability and warmup handling

Clear separation of retrieval, ranking, and generation

Vendor-agnostic, open-source deployment

The architecture closely aligns with systems used in search, recommendation, and enterprise QA pipelines at large technology companies.

---

### Tech Stack

Python

FAISS

BM25

Sentence-Transformers

Transformers (FLAN-T5)

FastAPI

Docker

---

### Author

Bhoomika Pathapati
MS in Data Science
