# pylint: disable=missing-function-docstring,redefined-outer-name

from typing import Iterable, Literal

from numpy import array_equal
from pytest import fixture

from server.dependencies import embedder_model
from server.features.embeddings import Embedder

type Text = Literal['Hello world!']


@fixture()
def embedding() -> Iterable[Embedder]:
    return embedder_model()


@fixture()
def text():
    yield 'Hello world!'


def test_encodings(embedding: Embedder, text: Text):
    assert array_equal(embedding.encode_query(text), embedding.encode_normalise(text)) is False


def test_encode_query(embedding: Embedder, text: Text):
    assert len(embedding.encode_query(text)) > 0


def test_encode_normalise(embedding: Embedder, text: Text):
    assert len(embedding.encode_normalise(text)) > 0
