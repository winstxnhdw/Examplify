from contextlib import asynccontextmanager
from typing import AsyncIterator

from litestar import Litestar
from redis.asyncio import Redis

from server.config import Config
from server.databases.redis import RedisAsync


@asynccontextmanager
async def create_redis_index(app: Litestar) -> AsyncIterator[None]:
    """
    Summary
    -------
    initialise a Redis index

    Parameters
    ----------
    app (Litestar) : the Litestar application
    """
    async with Redis.from_pool(app.state.redis_pool) as client:
        await RedisAsync(client).create_index(
            Config.redis_index_name,
            Config.embedding_dimensions,
            Config.document_index_tag,
            [Config.document_index_prefix],
        )

    try:
        yield

    finally:
        pass
