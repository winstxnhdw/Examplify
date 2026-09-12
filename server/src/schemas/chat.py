from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

from msgspec import Meta, Struct, field
from uuid_utils.compat import uuid7

from src.features.chat import Role


class Message(Struct, kw_only=True):
    id: Annotated[UUID, Meta(examples=["0195f17f-f01a-7347-be7e-9b5074881c2c"])]
    role: Annotated[Role, Meta(examples=["user", "assistant", "system"])]
    chat_id: Annotated[UUID, Meta(examples=["0195f17ff01a7347be7e9b5074881c2c"])]
    text: Annotated[str | None, Meta(examples=["How do I uninstall Java?"])]


class Chat(Struct, kw_only=True):
    id: Annotated[UUID, Meta(examples=["0195f17f-f01a-7347-be7e-9b5074881c2c"])] = field(default_factory=uuid7)
    title: Annotated[str, Meta(examples=["How do I uninstall Java?"])]
    messages: Sequence[Message] = []
