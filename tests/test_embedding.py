# pylint: disable=missing-function-docstring,redefined-outer-name

from typing import Iterator, Literal

from numpy import array_equal
from pytest import fixture

from server.features.embeddings import Embedder

type Text = Literal['Hello world!']


@fixture()
def embedding() -> Iterator[Embedder]:
    yield Embedder()


@fixture()
def text() -> Iterator[Text]:
    yield 'Hello world!'


def test_encodings(embedding: Embedder, text: Text):
    assert array_equal(embedding.encode_query(text), embedding.encode_normalise(text)) is False


def test_encode_query(embedding: Embedder, text: Text):
    assert len(embedding.encode_query(text)) > 0


def test_encode_normalise(embedding: Embedder, text: Text):
    assert len(embedding.encode_normalise(text)) > 0
