from src.config import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2"}
MIN_WORDS = 25
MIN_RATING = 4


def embeddings_path(model_key):
    return PROCESSED_DIR / f"embeddings_{model_key}.npz"