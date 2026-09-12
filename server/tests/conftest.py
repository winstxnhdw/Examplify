from typing import Literal

from pytest import fixture


@fixture
def anyio_backend() -> tuple[Literal["asyncio", "trio"], dict[str, bool]]:
    return "asyncio", {"use_uvloop": True}


@fixture
def text() -> Literal["Hello, world!"]:
    return "Hello, world!"
