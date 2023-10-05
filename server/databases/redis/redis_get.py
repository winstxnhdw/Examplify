from json import loads
from typing import TypeVar

from redis.asyncio import Redis

T = TypeVar('T')

async def redis_get(redis: Redis, key: str, default: T) -> T:
    """
    Summary
    -------
    get a value from redis, or return a default value

    Parameters
    ----------
    redis (Redis): the async redis client
    key (str): the key to get
    default (T): the default value
    """
    value: bytes | None = await redis.get(key)

    return (
        default
        if not value
        else loads(value)  # type: ignore
    )
