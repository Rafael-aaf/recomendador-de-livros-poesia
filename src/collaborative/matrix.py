import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_positive_matrix(df_ratings):
    df_positive = df_ratings[df_ratings["is_positive"]]
    user_index = pd.Index(df_positive["user_id"].unique())
    book_index = pd.Index(df_ratings["book_id"].unique())

    rows = user_index.get_indexer(df_positive["user_id"])
    columns = book_index.get_indexer(df_positive["book_id"])
    data = np.ones(len(df_positive), dtype=np.float32)

    matrix = csr_matrix((data, (rows, columns)), shape=(len(user_index), len(book_index)))
    return matrix, user_index, book_index