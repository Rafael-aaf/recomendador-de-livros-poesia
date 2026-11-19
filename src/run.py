import argparse

import numpy as np

from src.collaborative.matrix import build_positive_matrix
from src.collaborative.recommender import recommend_for_csv
from src.config import DATA_DIR
from src.data_prep.dataset_prep import load_prepared_dataset
from src.data_prep.matching import match_user_books
from src.data_prep.user_data import load_user_csv
from src.embeddings.search import load_embeddings, user_vector
from src.embeddings.search import recommend as recommend_semantic
from src.embeddings.settings import MIN_RATING, DEFAULT_MODEL
from src.merge.merge import merge_candidates

INPUT_DIR = DATA_DIR / "input"
SOURCE_LABELS = {"collaborative": "filtro", "semantic": "busca", "both": "ambos"}


class Recommender:
    def __init__(self, df_books, matrix, book_index, ids, emb):
        self.df_books = df_books
        self.matrix = matrix
        self.book_index = book_index
        self.ids = ids
        self.emb = emb
        self.catalog = (
            df_books.assign(
                book_id=df_books["book_id"].astype(str),
                authors=df_books["authors"].apply(lambda a: a if isinstance(a, str) else ", ".join(map(str, a))),
            )
            .drop_duplicates("book_id")
            .set_index("book_id")[["title", "authors"]]
        )

    @classmethod
    def load(cls, model_key=DEFAULT_MODEL):
        df_books, df_ratings, _ = load_prepared_dataset()
        matrix, _, book_index = build_positive_matrix(df_ratings)
        ids, emb = load_embeddings(model_key)
        return cls(df_books, matrix, book_index, ids, emb)

    def recommend(self, df_user, n_collaborative=10, n_semantic=5):
        df_matched = match_user_books(df_user, self.df_books)
        matched = df_matched.dropna(subset=["dataset_book_id"])
        owned = set(matched["dataset_book_id"])

        df_cf = recommend_for_csv(df_matched, self.matrix, self.book_index)
        df_cf["book_id"] = df_cf["book_id"].astype(str)

        ratings = (
            matched.dropna(subset=["rating"])
            .drop_duplicates("dataset_book_id")
            .set_index("dataset_book_id")["rating"]
        )
        vec = user_vector(ratings, self.ids, self.emb, min_rating=MIN_RATING)
        if vec is None:
            df_sem = df_cf.head(0).assign(sem_score=[]).drop(columns="cf_score")
        else:
            df_sem = recommend_semantic(vec, self.ids, self.emb, exclude=owned, k=len(self.ids))
            # livros excluídos entram no fim do ranking com score -inf
            df_sem = df_sem[np.isfinite(df_sem["score"])].rename(columns={"score": "sem_score"})

        result = merge_candidates(df_cf, df_sem, n_collaborative, n_semantic)
        result = result.join(self.catalog, on="book_id")
        return result[["book_id", "title", "authors", "source", "cf_score", "sem_score"]]


def find_user_csv():
    files = list(INPUT_DIR.glob("*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"Esperado 1 CSV em {INPUT_DIR}, encontrado(s): {[f.name for f in files]}")
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-collaborative", type=int, default=10)
    parser.add_argument("--n-semantic", type=int, default=5)
    args = parser.parse_args()

    recommender = Recommender.load()
    df_user = load_user_csv(find_user_csv())
    result = recommender.recommend(df_user, args.n_collaborative, args.n_semantic)

    table = result[["title", "authors", "source", "cf_score", "sem_score"]].copy()
    table["source"] = table["source"].map(SOURCE_LABELS)
    table.columns = ["Título", "Autor", "Origem", "Score filtro", "Score busca"]
    table.index = range(1, len(table) + 1)
    print(table.to_string(na_rep="-", float_format="{:.3f}".format, max_colwidth=60))


if __name__ == "__main__":
    main()