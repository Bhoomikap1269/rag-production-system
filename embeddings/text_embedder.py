from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm

DATA_PATH = "data/processed/products.jsonl"
EMB_OUT = "data/processed/text_embeddings.npy"
IDS_OUT = "data/processed/text_ids.npy"

# Load model ONCE
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_corpus():
    texts = []
    ids = []

    with open(DATA_PATH, "r") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            ids.append(obj["id"])

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(EMB_OUT, embeddings.astype("float32"))
    np.save(IDS_OUT, np.array(ids))

    print("Embedding matrix shape:", embeddings.shape)

def embed_query(text: str):
    """
    Embed a single query string.
    Returns shape (1, dim)
    """
    emb = model.encode([text], normalize_embeddings=True)
    return emb.astype("float32")

if __name__ == "__main__":
    embed_corpus()
