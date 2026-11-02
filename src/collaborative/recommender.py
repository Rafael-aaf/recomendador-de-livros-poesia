import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data_prep.dataset_prep import POSITIVE_THRESHOLD

N_NEIGHBORS = 50


def build_user_vector(positive_book_ids, book_index):
    columns = book_index.get_indexer(positive_book_ids)
    columns = columns[columns >= 0]
    data = np.ones(len(columns), dtype=np.float32)
    rows = np.zeros(len(columns), dtype=np.int32)
    return csr_matrix((data, (rows, columns)), shape=(1, len(book_index)))


def compute_similarity(matrix, user_vector):
    overlap = matrix.dot(user_vector.T).toarray().ravel()
    # vetores binários: a norma é a raiz do número de 1s
    user_norms = np.sqrt(np.asarray(matrix.sum(axis=1)).ravel())
    vector_norm = np.sqrt(user_vector.sum())
    return overlap / (user_norms * vector_norm)


def recommend(positive_book_ids, owned_book_ids, matrix, book_index, n_neighbors=N_NEIGHBORS):
    user_vector = build_user_vector(positive_book_ids, book_index)
    if user_vector.nnz == 0:
        raise ValueError("Nenhum livro bem avaliado do CSV está disponível para a filtragem colaborativa")

    similarity = compute_similarity(matrix, user_vector)
    neighbor_rows = np.argsort(similarity)[::-1][:n_neighbors]
    neighbor_rows = neighbor_rows[similarity[neighbor_rows] > 0]

    scores = matrix[neighbor_rows].T.dot(similarity[neighbor_rows])
    owned_columns = book_index.get_indexer(owned_book_ids)
    scores[owned_columns[owned_columns >= 0]] = 0

    candidate_columns = np.flatnonzero(scores > 0)
    df_candidates = pd.DataFrame(
        {"book_id": book_index[candidate_columns], "cf_score": scores[candidate_columns]}
    )
    return df_candidates.sort_values("cf_score", ascending=False, ignore_index=True)


def recommend_for_csv(df_matched, matrix, book_index, n_neighbors=N_NEIGHBORS):
    is_positive = df_matched["rating"] >= POSITIVE_THRESHOLD
    positive_book_ids = df_matched.loc[is_positive, "dataset_book_id"].dropna().unique()
    owned_book_ids = df_matched["dataset_book_id"].dropna().unique()
    return recommend(positive_book_ids, owned_book_ids, matrix, book_index, n_neighbors)