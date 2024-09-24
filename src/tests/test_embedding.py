# pylint: disable=missing-function-docstring,redefined-outer-name

from typing import Iterator

from numpy import array_equal
from pytest import fixture

from examplify.features.embeddings import Embedder


@fixture
def embedding() -> Iterator[Embedder]:
    yield Embedder()


def test_encodings(embedding: Embedder, text: str):
    assert array_equal(embedding.encode_query(text), embedding.encode_normalise(text)) is False


def test_encode_query(embedding: Embedder, text: str):
    assert len(embedding.encode_query(text)) > 0


def test_encode_normalise(embedding: Embedder, text: str):
    assert len(embedding.encode_normalise(text)) > 0
