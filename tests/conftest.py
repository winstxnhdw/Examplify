# pylint: disable=missing-function-docstring

from typing import Literal

from pytest import fixture


@fixture
def anyio_backend() -> tuple[Literal['asyncio', 'trio'], dict[str, bool]]:
    return 'asyncio', {'use_uvloop': True}
