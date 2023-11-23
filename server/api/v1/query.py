from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis

from server.api.v1 import v1
from server.dependencies import get_redis_client
from server.features import query_llm
from server.schemas.v1 import Answer, Query


@v1.post('/{chat_id}/query')
async def query(
    redis: Annotated[Redis, Depends(get_redis_client)],
    chat_id: str,
    request: Query,
    top_k: int = 5,
    store_query: bool = True
) -> Answer:
    """
    Summary
    -------
    the `/query` route provides an endpoint for performning retrieval-augmented generation
    """
    messages = await query_llm(redis, request.query, chat_id, top_k, store_query)
    return Answer(messages=messages)
