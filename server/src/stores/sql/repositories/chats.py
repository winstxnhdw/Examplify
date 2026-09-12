from collections.abc import Sequence
from typing import Any

from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from sqlalchemy import select
from sqlalchemy.orm import QueryableAttribute, load_only

from src.stores.sql.models import ChatModel


class ChatStoreRepository(SQLAlchemyAsyncRepository[ChatModel]):
    model_type = ChatModel

    async def list_chats(
        self,
        load_fields: Sequence[QueryableAttribute[Any]] | None = None,
    ) -> tuple[Sequence[ChatModel], int]:
        return await self.list_and_count(
            statement=None if not load_fields else select(ChatModel).options(load_only(*load_fields)),
        )
