import json

DATA_PATH = "data/processed/products.jsonl"

DOC_STORE = {}

with open(DATA_PATH, "r") as f:
    for line in f:
        obj = json.loads(line)
        DOC_STORE[obj["id"]] = obj["text"]

def get_text(doc_id: int) -> str:
    return DOC_STORE.get(doc_id, "")

