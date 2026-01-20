import json
from rank_bm25 import BM25Okapi
import pickle

DATA_PATH = "data/processed/products.jsonl"
INDEX_PATH = "vector_store/bm25.pkl"

def tokenize(text):
    return text.lower().split()

def build_bm25():
    corpus = []
    documents = []

    with open(DATA_PATH, "r") as f:
        for line in f:
            obj = json.loads(line)
            documents.append(obj)
            corpus.append(tokenize(obj["text"]))

    bm25 = BM25Okapi(corpus)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump((bm25, documents), f)

    print("BM25 index built")
    print(f"Documents: {len(documents)}")

if __name__ == "__main__":
    build_bm25()
