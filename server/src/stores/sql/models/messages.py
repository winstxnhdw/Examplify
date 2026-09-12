from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid_utils.compat import uuid7

from src.features.chat import Role
from src.stores.sql.models.base import Base
from src.stores.sql.models.chats import ChatModel


class MessageModel(Base):
    __tablename__ = "messages"

    role: Mapped[Role]
    text: Mapped[str | None] = mapped_column(default=None)
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid7, init=False)
    chat_id: Mapped[UUID] = mapped_column(ForeignKey("chats.id", ondelete="cascade"), init=False)
    chat: Mapped[ChatModel] = relationship(
        back_populates=ChatModel.messages.key,
        foreign_keys="MessageModel.chat_id",
        init=False,
    )
