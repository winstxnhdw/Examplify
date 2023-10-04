from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis

from server.api.v1 import v1
from server.config import Config
from server.databases.redis import create_query_parameters
from server.databases.redis import redis_query as redis_query_helper
from server.dependencies import get_redis_client
from server.features import Embedding
from server.schemas.v1 import Search


@v1.post('/{chat_id}/search')
async def search(
    chat_id: str,
    request: Search,
    redis: Annotated[Redis, Depends(get_redis_client)]
) -> str:
    """
    Summary
    -------
    the `/search` route provides an endpoint for searching the vector database
    """
    redis_query_parameters = create_query_parameters(
        Embedding().encode_query(request.query)
    )

    search_response = await redis.ft(Config.redis_index_name).search(
        redis_query_helper('tag', chat_id, request.top_k),
        redis_query_parameters  # type: ignore  (this is a bug in the redis-py library)
    )

    return search_response.docs[0]['content']  # type: ignore
