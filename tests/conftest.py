# pylint: disable=missing-function-docstring

from pytest import fixture
from uvloop import EventLoopPolicy


@fixture(scope="session")
def event_loop_policy():

    return EventLoopPolicy()
