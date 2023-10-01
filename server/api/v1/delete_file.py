
from typing import Annotated

from fastapi import Depends
from redis import Redis

from server.api.v1 import v1
from server.dependencies import get_redis_client


@v1.get('/delete_file/{file_id}')
def delete_file(
    file_id: str,
    redis: Annotated[Redis, Depends(get_redis_client)],
):
    """
    Summary
    -------
    the `/delete_file` route provides an endpoint for deleting a file from the Redis database
    """
    redis.delete(file_id)
