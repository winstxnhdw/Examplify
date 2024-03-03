from typing import Annotated

from fastapi import Depends

from server.api.v1 import v1
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.schemas.v1 import Timestamp


@v1.delete('/{chat_id}/delete_file/{file_id}')
async def delete_file(
    chat_id: str,
    file_id: str,
    redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)],
) -> Timestamp:
    """
    Summary
    -------
    the `/delete_file` route provides an endpoint for deleting a file
    """
    await redis.delete_document(chat_id, file_id)
    return Timestamp()
