from typing import Annotated

from fastapi import Depends

from server.api.v1 import v1
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.schemas.v1 import Timestamp


@v1.delete('/{chat_id}/delete_chat')
async def delete_chat(
    chat_id: str,
    redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)],
) -> Timestamp:
    """
    Summary
    -------
    the `/delete_chat` route provides an endpoint for deleting a chat
    """
    await redis.delete_chat(chat_id)
    return Timestamp()
