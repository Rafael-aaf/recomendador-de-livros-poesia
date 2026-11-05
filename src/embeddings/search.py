import numpy as np
import pandas as pd

from src.embeddings.settings import MIN_RATING, embeddings_path


def load_embeddings(model_key="minilm"):
    data = np.load(embeddings_path(model_key), allow_pickle=True)
    return data["ids"].astype(str), data["emb"]


def user_vector(ratings, ids, emb, min_rating=MIN_RATING):
    liked = ratings[ratings >= min_rating]
    pos = pd.Index(ids).get_indexer(liked.index)
    keep = pos >= 0
    if not keep.any():
        return None
    weights = liked.to_numpy(dtype=np.float32)[keep]
    vec = weights @ emb[pos[keep]]
    return vec / np.linalg.norm(vec)


def recommend(vec, ids, emb, exclude=(), k=50):
    scores = emb @ vec
    if exclude:
        scores[np.isin(ids, list(exclude))] = -np.inf
    top = np.argsort(-scores)[:k]
    return pd.DataFrame({"book_id": ids[top], "score": scores[top]})