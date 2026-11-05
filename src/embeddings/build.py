import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import BOOKS_PATH
from src.embeddings.settings import MODELS, MIN_WORDS, PROCESSED_DIR, embeddings_path


def load_valid_books():
    books = pd.read_json(BOOKS_PATH, lines=True)
    words = books["description"].fillna("").str.split().str.len()
    books = books[words >= MIN_WORDS]
    return books["book_id"].astype(str).to_numpy(dtype=str), books["description"].tolist()


def main(model_key="minilm"):
    ids, texts = load_valid_books()
    print(f"{len(ids):,} livros com sinopse válida")
    model = SentenceTransformer(MODELS[model_key])
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(embeddings_path(model_key), ids=ids, emb=emb.astype(np.float32))


if __name__ == "__main__":
    main(*sys.argv[1:])