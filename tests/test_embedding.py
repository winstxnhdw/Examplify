# pylint: disable=missing-function-docstring,redefined-outer-name

from typing import Literal

from numpy import array_equal
from pytest import fixture

from server.features.embeddings import Embedding

type Text = Literal['Hello world!']


@fixture()
def embedding():
    yield Embedding(force_download=True)


@fixture()
def text():
    yield 'Hello world!'


def test_encodings(embedding: Embedding, text: Text):
    assert array_equal(embedding.encode_query(text), embedding.encode_normalise(text)) is False


def test_encode_query(embedding: Embedding, text: Text):
    assert len(embedding.encode_query(text)) > 0


def test_encode_normalise(embedding: Embedding, text: Text):
    assert len(embedding.encode_normalise(text)) > 0
