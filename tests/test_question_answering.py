# pylint: disable=missing-function-docstring

from typing import Literal, Sequence

from pytest import fixture, mark

from server.features.llm.types import Message
from server.features.question_answering.question_answering import question_answering


@fixture
def anyio_backend() -> tuple[Literal['asyncio', 'trio'], dict[str, bool]]:
    return ('asyncio', {'use_uvloop': True})


async def chain(messages: Sequence[Message]) -> Message | None:

    if len(messages) > 1:
        return None

    return {'role': 'assistant', 'content': 'Hello world!'}

@mark.anyio
async def test_question_answering():

    messages: list[Message] = [
        {'role': 'assistant', 'content': 'Hello world!'},
        {'role': 'user', 'content': 'Hello world!'}
    ]

    answers = await question_answering('', '', messages, chain)

    assert len(answers) == 2
    assert answers[-1]['role'] == 'assistant'
