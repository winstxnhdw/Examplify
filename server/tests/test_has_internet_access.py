# ruff: noqa: S101

from pytest import mark
from src.utils.network.has_internet_access import has_internet_access


@mark.parametrize(
    "repository",
    [
        "winstxnhdw/openchat-3.6-ct2-int8",
        "winstxnhdw/bge-base-en-v1.5-ct2",
    ],
)
def test_has_internet_access(repository: str) -> None:
    assert has_internet_access(repository)
