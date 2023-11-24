from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis

from server.api.v1 import v1
from server.databases.redis.features import save_messages, search
from server.databases.redis.helpers import redis_get
from server.dependencies import get_redis_client
from server.features import Embedding, query_llm
from server.features.llm.types import Message
from server.schemas.v1 import Answer, Query


@v1.post('/{chat_id}/query')
async def query(
    redis: Annotated[Redis, Depends(get_redis_client)],
    chat_id: str,
    request: Query,
    top_k: int = 1,
    store_query: bool = True
) -> Answer:
    """
    Summary
    -------
    the `/query` route provides an endpoint for performning retrieval-augmented generation
    """
    embedding = Embedding().encode_query(request.query)
    context = await search(redis, chat_id, embedding, top_k)
    message_history: list[Message] = await redis_get(redis, f'chat:{chat_id}', _ := [])
    save_messages_middleware = save_messages(redis, chat_id, store_query)

    messages = await query_llm(
        request.query,
        context,
        message_history,
        save_messages_middleware
    )

    return Answer(messages=messages)
