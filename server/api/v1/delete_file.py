
from typing import Annotated

from fastapi import Depends
from redis import Redis

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client


@v1.get(f'/delete_file/{file_id}', deprecated=True)
async def delete_file(
    file_id: str,
    redis: Annotated[Redis, Depends(get_redis_client)],
):
    """
    Summary
    -------
    the `/delete_file` route provides an endpoint for deleting a file from the Redis database
    """
    return await redis.delete(f'{Config.document_index_prefix}:{file_id}')
