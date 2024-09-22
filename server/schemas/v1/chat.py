from typing import Annotated

from fastnanoid import generate
from msgspec import Meta, Struct, field


class Chat(Struct):
    """
    Summary
    -------
    the chat response schema

    Attributes
    ----------
    id (str) : the chat id
    """

    id: Annotated[str, Meta(examples=['glsODUdnaUei_A_IxfSRI'])] = field(default_factory=generate)
