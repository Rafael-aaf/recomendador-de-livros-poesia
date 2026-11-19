def summarize(df_hits):
    df = df_hits.assign(
        recall=df_hits["hits"] / df_hits["n_hidden"],
        hit_rate=(df_hits["hits"] > 0).astype(float),
    )
    return df.groupby(["k", "method"], sort=False)[["recall", "hit_rate"]].mean()