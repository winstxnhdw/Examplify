from typing import Annotated

from fastapi import Depends

from server.api.debug import debug
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.schemas.v1 import Timestamp


@debug.delete('/delete_all')
async def delete_all(redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)]) -> Timestamp:
    """
    Summary
    -------
    the `/delete_all` route provides an endpoint for deleting all files from the Redis database
    """
    await redis.recreate_index()
    return Timestamp()
