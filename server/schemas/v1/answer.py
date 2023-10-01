from collections import deque

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
    messages: deque[Message] = Field(examples=[[
        {
            'role': 'user',
            'content': 'What is 1 + 1?'
        },
        {
            'role': 'assistant',
            'content': '2.'
        }
    ]])
