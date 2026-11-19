import numpy as np
import pandas as pd


def split_holdout(df_ratings, n_users, n_hidden, min_positives, seed):
    rng = np.random.default_rng(seed)
    positives = df_ratings[df_ratings["is_positive"]]
    counts = positives.groupby("user_id").size()
    eligible = counts[counts >= min_positives].index.to_numpy()
    test_users = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)

    hidden_rows = (
        positives[positives["user_id"].isin(test_users)]
        .groupby("user_id", group_keys=False)
        .sample(n=n_hidden, random_state=seed)
    )
    hidden = hidden_rows.groupby("user_id")["book_id"].apply(set).to_dict()

    hidden_keys = pd.MultiIndex.from_frame(hidden_rows[["user_id", "book_id"]])
    all_keys = pd.MultiIndex.from_frame(df_ratings[["user_id", "book_id"]])
    train = df_ratings[~all_keys.isin(hidden_keys)]
    return train, hidden