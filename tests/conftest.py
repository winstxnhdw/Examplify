# pylint: disable=missing-function-docstring

from typing import Literal

from pytest import fixture
from uvloop import EventLoopPolicy


@fixture(scope='session')
def event_loop_policy():
    return EventLoopPolicy()


@fixture
def anyio_backend() -> tuple[Literal['asyncio', 'trio'], dict[str, bool]]:
    return 'asyncio', {'use_uvloop': True}
