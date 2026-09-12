from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from src.stores.sql.services import ChatStoreService


async def chat_store_service(db_session: AsyncSession) -> AsyncIterator[ChatStoreService]:
    async with ChatStoreService.new(session=db_session) as service:
        yield service
