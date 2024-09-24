# pylint: disable=missing-function-docstring

from typing import Iterator

from examplify.features.chat.types import Message
from examplify.features.question_answering.question_answering import question_answering


def chain(messages: list[Message]) -> Iterator[str] | None:
    if len(messages) > 1:
        return None

    return (message['content'] for message in messages)


def test_question_answering(text: str):
    messages: list[Message] = [
        {'role': 'assistant', 'content': text},
        {'role': 'user', 'content': text},
    ]

    assert next(question_answering(messages, chain)) == text
