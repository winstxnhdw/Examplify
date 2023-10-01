
from typing import Annotated

from fastapi import Depends
from redis import Redis

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client
from server.lifespans import create_redis_index


@v1.get('/delete_all')
def delete_all(
    redis: Annotated[Redis, Depends(get_redis_client)],
):
    """
    Summary
    -------
    the `/delete_all` route provides an endpoint for deleting all files from the Redis database
    """
    redis.ft(Config.redis_index_name).dropindex(delete_documents=True)
    create_redis_index()
