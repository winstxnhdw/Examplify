from pydantic import BaseModel, Field

from server.features.llm.types import Message


class Answer(BaseModel):
    """
    Summary
    -------
    the response to a query

    Attributes
    ----------
    answer (str) : the answer
    """

    messages: list[Message] = Field(
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
    )
