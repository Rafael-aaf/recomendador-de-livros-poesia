import pandas as pd


def match_user_books(df_user, df_books):
    df = df_user.copy()

    # o Goodreads exporta o ISBN como ="978..." para o Excel não converter em número
    user_isbn13 = df["ISBN13"].str.strip('="')
    user_isbn13 = user_isbn13.where(user_isbn13 != "")

    dataset_isbn13 = df_books["isbn13"].astype(str).str.strip()
    has_isbn13 = dataset_isbn13 != ""
    isbn13_to_book_id = pd.Series(
        df_books.loc[has_isbn13, "book_id"].values,
        index=dataset_isbn13[has_isbn13].values,
    )

    by_book_id = df["Book Id"].where(df["Book Id"].isin(set(df_books["book_id"])))
    by_isbn13 = user_isbn13.map(isbn13_to_book_id)

    df["dataset_book_id"] = by_book_id.fillna(by_isbn13)
    df["match_source"] = None
    df.loc[by_isbn13.notna(), "match_source"] = "isbn13"
    df.loc[by_book_id.notna(), "match_source"] = "book_id"
    return df


def print_match_summary(df_matched):
    total = len(df_matched)
    matched = df_matched["match_source"].notna()
    by_source = df_matched["match_source"].value_counts()
    rated = df_matched["rating"].notna()

    print(f"Livros no CSV: {total}")
    print(f"Casados por Book Id: {by_source.get('book_id', 0)}")
    print(f"Casados por ISBN13 (não casaram por Book Id): {by_source.get('isbn13', 0)}")
    print(f"Não casados: {(~matched).sum()}")
    print(f"Taxa de casamento: {matched.mean():.1%}")
    print(f"Livros com nota que casaram: {(rated & matched).sum()} de {rated.sum()}")