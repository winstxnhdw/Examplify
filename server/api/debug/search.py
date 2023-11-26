from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis

from server.api.debug import debug
from server.databases.redis.features import search as redis_search
from server.dependencies import get_redis_client
from server.features import Embedding
from server.schemas.v1 import Query


@debug.post('/{chat_id}/search')
async def search(
    redis: Annotated[Redis, Depends(get_redis_client)],
    chat_id: str,
    request: Query,
    top_k: int = 5
) -> str:
    """
    Summary
    -------
    the `/search` route provides an endpoint for searching the vector database
    """
    embedding = Embedding().encode_query(request.query)
    return await redis_search(redis, chat_id, embedding, top_k)
