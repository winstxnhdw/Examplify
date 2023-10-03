from collections import deque

from pydantic import Field

from server.features.llm.types import Message
from server.schemas.v1.base import Search


class Query(Search):
    """
    Summary
    -------
    the query request

    Attributes
    ----------
    messages (deque[Message]) : the list of previous messages
    """
    messages: deque[Message] = Field(examples=[[
        {
            'role': 'user',
            'content': 'What is 1 + 1?'
        },
        {
            'role': 'assistant',
            'content': '2.'
        },
    ]], default=deque())
