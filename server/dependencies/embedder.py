from typing import Iterator

from server.features.embeddings import Embedder


def embedder() -> Iterator[Embedder]:
    """
    Summary
    -------
    load the embeddings model

    Returns
    -------
    embedding (Embedding): the embeddings model
    """
    embedder = Embedder()

    try:
        yield embedder

    finally:
        del embedder
