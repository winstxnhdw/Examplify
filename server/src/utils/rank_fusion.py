from collections.abc import Hashable, Iterable, Iterator
from itertools import chain

from numpy import argsort, dtype, float32, full, ndarray
from numpy import sum as np_sum


def rank_fusion[T: Hashable](*documents_args: Iterable[T], alpha: int = 60) -> Iterator[T]:
    unique_documents_dict = {document: i for i, document in enumerate(set(chain.from_iterable(documents_args)))}
    unique_documents_list = list(unique_documents_dict)
    unique_documents_count = len(unique_documents_list)
    rank_matrix: ndarray[tuple[int, int], dtype[float32]] = full(
        (unique_documents_count, len(documents_args)),
        unique_documents_count,
        dtype=float32,
    )

    for ranker, documents in enumerate(documents_args):
        for rank, document in enumerate(documents):
            rank_matrix[unique_documents_dict[document], ranker] = rank

    return (unique_documents_list[i] for i in argsort(-np_sum(1 / (alpha + rank_matrix), axis=1)))
