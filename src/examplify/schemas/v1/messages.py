from typing import Annotated

from msgspec import Meta, Struct

from examplify.features.chat.types import Message


class Messages(Struct):
    """
    Summary
    -------
    the message history schema

    Attributes
    ----------
    messages (list[Message]) : the message history
    """

    messages: Annotated[
        list[Message],
        Meta(
            examples=[
                [
                    {
                        'role': 'user',
                        'content': 'What is the capital of Japan?',
                    },
                    {
                        'role': 'assistant',
                        'content': 'Tokyo.',
                    },
                ]
            ]
        ),
    ]
