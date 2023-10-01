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
    messages (list[Message]) : the list of previous messages
    """
    messages: list[Message] = Field(examples=[[
        {
            'role': 'user',
            'content': 'You are a helpful AI assistant. You are given the following chat history. Answer the question based on the context provided as truthfully as you are able to. If you do not know the answer, you may respond with "I do not know". What is the Baloney Detection Kit?'
        },
        {
            'role': 'assistant',
            'content': 'The Baloney Detection Kit is a a set of cognitive tools and techniques that fortify the mind against penetration by falsehoods. It was created by Carl Sagan.'
        },
    ]])
