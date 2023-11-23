from typing import Literal

from typing_extensions import TypedDict


class Message(TypedDict):
    """
    Summary
    -------
    an LLM message

    Attributes
    ----------
    role (Literal['user', 'assistant']) : the role of the message
    content (str) : the content of the message
    """
    role: Literal['user', 'assistant', 'system']
    content: str
