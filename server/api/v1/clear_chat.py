from typing import Annotated

from fastapi import Depends

from server.api.v1 import v1
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.schemas.v1 import Timestamp


@v1.delete('/{chat_id}/clear_chat')
async def clear_chat(
    chat_id: str,
    redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)],
) -> Timestamp:
    """
    Summary
    -------
    the `/clear_chat` route provides an endpoint to clear all chat history
    """
    await redis.delete_chat_messages(chat_id)
    return Timestamp()
