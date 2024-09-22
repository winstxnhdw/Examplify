from typing import AsyncIterator

from redis.asyncio import Redis

from server.databases.redis import RedisAsync
from server.state import AppState


async def redis_client(state: AppState) -> AsyncIterator[RedisAsync]:
    """
    Summary
    -------
    provides an async Redis client from the connection pool

    Yields
    ------
    client (RedisAsyncWrapper) : a Redis client
    """
    async with Redis.from_pool(state.redis_pool) as redis:
        yield RedisAsync(redis)
