from pathlib import Path

import pandas as pd

from src.config import AUTHORS_PATH, BOOKS_PATH, INTERACTIONS_PATH

POSITIVE_THRESHOLD = 4
MIN_RATINGS = 5
MIN_DESCRIPTION_WORDS = 25
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_author_names():
    df = pd.read_json(AUTHORS_PATH, lines=True, convert_dates=False, dtype={"author_id": str})
    return df.set_index("author_id")["name"].to_dict()


def load_books():
    df = pd.read_json(
        BOOKS_PATH,
        lines=True,
        convert_dates=False,
        dtype={"book_id": str, "isbn13": str, "publication_year": str},
    )
    author_names = load_author_names()
    df["authors"] = df["authors"].apply(
        lambda authors: ", ".join(author_names.get(str(a["author_id"]), str(a["author_id"])) for a in authors)
    )
    df = df[["book_id", "title", "authors", "publication_year", "description", "isbn13"]].copy()
    df["book_id"] = df["book_id"].astype(str)
    return df


def load_interactions():
    df = pd.read_json(INTERACTIONS_PATH, lines=True, convert_dates=False, dtype={"user_id": str, "book_id": str})
    df = df[["user_id", "book_id", "rating"]].copy()
    df["book_id"] = df["book_id"].astype(str)
    return df


def filter_min_ratings(df, min_ratings=MIN_RATINGS):
    # remover usuários muda a contagem dos livros e vice-versa, então repete até nada mais sair
    while True:
        n_before = len(df)
        df = df[df.groupby("user_id")["book_id"].transform("size") >= min_ratings]
        df = df[df.groupby("book_id")["user_id"].transform("size") >= min_ratings]
        if len(df) == n_before:
            return df


def select_semantic_books(df_books):
    word_count = df_books["description"].fillna("").str.split().str.len()
    return df_books[word_count >= MIN_DESCRIPTION_WORDS]


def save_dataset(df_books, df_ratings, df_semantic_books):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_books.to_parquet(PROCESSED_DIR / "books.parquet", index=False)
    df_ratings.to_parquet(PROCESSED_DIR / "ratings.parquet", index=False)
    df_semantic_books.to_parquet(PROCESSED_DIR / "semantic_books.parquet", index=False)


def load_prepared_dataset():
    df_books = pd.read_parquet(PROCESSED_DIR / "books.parquet")
    df_ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    df_semantic_books = pd.read_parquet(PROCESSED_DIR / "semantic_books.parquet")
    return df_books, df_ratings, df_semantic_books


def prepare_dataset():
    df_books = load_books()
    df_interactions = load_interactions()

    df_ratings = df_interactions[df_interactions["rating"] > 0].copy()
    df_ratings["is_positive"] = df_ratings["rating"] >= POSITIVE_THRESHOLD
    df_ratings = filter_min_ratings(df_ratings)

    df_semantic_books = select_semantic_books(df_books)
    save_dataset(df_books, df_ratings, df_semantic_books)
    return df_books, df_ratings, df_semantic_books