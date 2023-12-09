from typing import AsyncGenerator

from redis.asyncio import Redis as RedisAsync

from server.databases import Redis
from server.databases.redis.wrapper import RedisAsyncWrapper


async def get_redis_client() -> AsyncGenerator[RedisAsyncWrapper, None]:
    """
    Summary
    -------
    get an async Redis client from the connection pool

    Yields
    ------
    client (RedisAsync) : a Redis client
    """
    async with RedisAsync.from_pool(Redis.pool) as redis:
        yield RedisAsyncWrapper(redis)
