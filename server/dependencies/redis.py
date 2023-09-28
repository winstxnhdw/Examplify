from typing import Generator

from redis import Redis as RedisProtocol

from server.databases import Redis


def get_redis_client() -> Generator[RedisProtocol, None, None]:
    """
    Summary
    -------
    get a Redis client from the connection pool

    Yields
    ------
    client (RedisProtocol) : a Redis client
    """
    with RedisProtocol.from_pool(Redis.pool) as redis:
        yield redis
