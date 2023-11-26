from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client
from server.lifespans import create_redis_index
from server.schemas.v1 import Timestamp


@v1.get('/delete_all')
async def delete_all(redis: Annotated[Redis, Depends(get_redis_client)]) -> Timestamp:
    """
    Summary
    -------
    the `/delete_all` route provides an endpoint for deleting all files from the Redis database
    """
    await redis.ft(Config.redis_index_name).dropindex(delete_documents=True)
    await create_redis_index()

    return Timestamp()
