import argparse
from itertools import islice

import numpy as np
import pandas as pd

from src.collaborative.matrix import build_positive_matrix
from src.collaborative.recommender import recommend as recommend_collaborative
from src.config import DATA_DIR
from src.data_prep.dataset_prep import load_prepared_dataset
from src.embeddings.search import load_embeddings, user_vector
from src.embeddings.search import recommend as recommend_semantic
from src.embeddings.settings import MIN_RATING
from src.evaluation.metrics import summarize
from src.evaluation.split import split_holdout
from src.merge.merge import merge_candidates

SEMANTIC_POOL = 200
REQUIRED_COLUMNS = {"user_id", "book_id", "rating", "is_positive"}
CF_SHARES = {"collaborative": 1.0, "merged 67/33": 2 / 3, "merged 50/50": 0.5, "semantic": 0.0}


def collaborative_candidates(user_rows, owned, matrix, book_index):
    positive_ids = user_rows.loc[user_rows["is_positive"], "book_id"].to_numpy()
    return recommend_collaborative(positive_ids, list(owned), matrix, book_index)


def semantic_candidates(user_rows, owned, ids, emb):
    ratings = user_rows.drop_duplicates("book_id").set_index("book_id")["rating"]
    vec = user_vector(ratings, ids, emb, min_rating=MIN_RATING)
    if vec is None:
        return pd.DataFrame({"book_id": [], "sem_score": []})
    df_sem = recommend_semantic(vec, ids, emb, exclude=owned, k=SEMANTIC_POOL)
    df_sem = df_sem[np.isfinite(df_sem["score"])]
    return df_sem.rename(columns={"score": "sem_score"})


def evaluate(model_key, n_users, n_hidden, min_positives, ks, seed):
    _, df_ratings, _ = load_prepared_dataset()
    missing = REQUIRED_COLUMNS - set(df_ratings.columns)
    if missing:
        raise KeyError(f"df_ratings sem as colunas: {sorted(missing)}")
    df_ratings = df_ratings.assign(book_id=df_ratings["book_id"].astype(str))

    train, hidden = split_holdout(df_ratings, n_users, n_hidden, min_positives, seed)
    # os livros escondidos saem da matriz inteira, senão o usuário vota neles pelos próprios dados
    matrix, _, book_index = build_positive_matrix(train)
    ids, emb = load_embeddings(model_key)

    hidden_all = list({b for books in hidden.values() for b in books})
    print(f"Usuários de teste: {len(hidden)}, livros escondidos: {len(hidden_all)}")
    print(f"Escondidos presentes na matriz do filtro: {pd.Index(hidden_all).isin(book_index).mean():.1%}")
    print(f"Escondidos com embedding (alcançáveis pela busca): {np.isin(hidden_all, ids).mean():.1%}")

    popularity = (
        train[train["is_positive"]].groupby("book_id").size().sort_values(ascending=False).index.tolist()
    )
    train_by_user = {u: g for u, g in train[train["user_id"].isin(hidden)].groupby("user_id")}

    rows = []
    for i, (user_id, hidden_books) in enumerate(hidden.items(), start=1):
        user_rows = train_by_user[user_id]
        owned = set(user_rows["book_id"])
        df_cf = collaborative_candidates(user_rows, owned, matrix, book_index)
        df_sem = semantic_candidates(user_rows, owned, ids, emb)

        for k in ks:
            recs = {"popularity": list(islice((b for b in popularity if b not in owned), k))}
            for method, share in CF_SHARES.items():
                n_cf = round(k * share)
                recs[method] = merge_candidates(df_cf, df_sem, n_cf, k - n_cf)["book_id"].tolist()
            for method, book_ids in recs.items():
                rows.append(
                    {
                        "user_id": user_id,
                        "k": k,
                        "method": method,
                        "hits": len(hidden_books & set(book_ids)),
                        "n_hidden": len(hidden_books),
                    }
                )
        if i % 100 == 0:
            print(f"{i}/{len(hidden)} usuários avaliados")

    return summarize(pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="minilm")
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--min-positives", type=int, default=10)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = evaluate(args.model, args.n_users, args.n_hidden, args.min_positives, args.k, args.seed)
    print()
    print(summary.round(4).to_string())

    output = DATA_DIR / "processed" / f"evaluation_{args.model}.csv"
    summary.assign(model=args.model).to_csv(output)
    print(f"\nSalvo em: {output}")


if __name__ == "__main__":
    main()