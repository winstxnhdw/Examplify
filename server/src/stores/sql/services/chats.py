from collections import deque
from collections.abc import AsyncIterator
from typing import Iterator
from uuid import UUID

from advanced_alchemy.exceptions import NotFoundError
from advanced_alchemy.filters import LimitOffset
from advanced_alchemy.service import OffsetPagination, SQLAlchemyAsyncRepositoryService

from src.features.answering import queue_answering
from src.features.chat import ChatAgentProtocol, Prompt, Prompts
from src.schemas import Chat
from src.stores.sql.models import ChatModel, MessageModel
from src.stores.sql.repositories import ChatStoreRepository


class ChatStoreService(SQLAlchemyAsyncRepositoryService[ChatModel, ChatStoreRepository]):
    repository_type = ChatStoreRepository

    async def get_chat(self, chat_id: UUID) -> Chat:
        return self.to_schema(await self.get(chat_id), schema_type=Chat)

    async def get_chats(self, limit_offset: LimitOffset, *, only_indices: bool) -> OffsetPagination[Chat]:
        results, total = await self.list_and_count(load=ChatModel.id if only_indices else None)
        return self.to_schema(results, total, (limit_offset,), schema_type=Chat)

    async def delete_chat(self, chat_id: UUID) -> NotFoundError | None:
        try:
            await self.delete(chat_id)

        except NotFoundError as error:
            return error

    async def answer_query(
        self,
        chat_agent: ChatAgentProtocol,
        chat_id: UUID,
        query: str,
        *,
        store_query: bool,
    ) -> Iterator[str]:
        chat = Chat(id=chat_id, title=query[:16], messages=[])
        # chat, _ = await self.get_or_upsert(match_fields="id", id=chat_id, title=query[:16])
        prompts = Prompts(Prompt(role=message.role, content=message.text) for message in chat.messages if message.text)
        prompts.append(Prompt(role="user", content=query))
        accumulator: deque[str] = deque()

        try:
            for item in queue_answering(chat_agent, prompts):
                accumulator.append(item)
                yield item

        finally:
            if store_query:
                prompts_to_store = [
                    MessageModel(role="user", text=query),
                    MessageModel(role="assistant", text="".join(accumulator)),
                ]

                chat.messages.extend(prompts_to_store)
                await self.update(chat)
                await self.repository.session.commit()
