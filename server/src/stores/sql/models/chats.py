from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid_utils.compat import uuid7

from src.stores.sql.models.base import Base

if TYPE_CHECKING:
    from src.stores.sql.models import MessageModel


class ChatModel(Base):
    __tablename__ = "chats"

    title: Mapped[str]
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid7)
    messages: Mapped[list[MessageModel]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
        default_factory=list,
    )
