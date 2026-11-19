from itertools import zip_longest

import pandas as pd


def merge_candidates(df_cf, df_sem, n_collaborative, n_semantic):
    cf_top = df_cf.head(n_collaborative)["book_id"].tolist()
    cf_ids = set(cf_top)

    sem_picks, both_ids = [], set()
    for book_id in df_sem["book_id"]:
        if len(sem_picks) == n_semantic:
            break
        if book_id in cf_ids:
            both_ids.add(book_id)
        else:
            sem_picks.append(book_id)

    both = [b for b in cf_top if b in both_ids]
    cf_only = [b for b in cf_top if b not in both_ids]
    interleaved = [b for pair in zip_longest(cf_only, sem_picks) for b in pair if b is not None]

    source = {b: "collaborative" for b in cf_only}
    source.update({b: "semantic" for b in sem_picks})
    source.update({b: "both" for b in both})

    result = pd.DataFrame({"book_id": both + interleaved})
    result["source"] = result["book_id"].map(source)
    result["cf_score"] = result["book_id"].map(df_cf.set_index("book_id")["cf_score"])
    result["sem_score"] = result["book_id"].map(df_sem.set_index("book_id")["sem_score"])
    return result