from typing import Annotated

from fastapi import Depends
from starlette.responses import PlainTextResponse

from server.api.debug import debug
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.features import Embedding
from server.schemas.v1 import Query


@debug.post('/{chat_id}/search')
async def search(
    redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)],
    chat_id: str,
    request: Query,
    top_k: int = 5,
) -> PlainTextResponse:
    """
    Summary
    -------
    the `/search` route provides an endpoint for searching the vector database
    """
    embedding = Embedding().encode_query(request.query)
    result = await redis.search(chat_id, embedding, top_k)

    return PlainTextResponse(result)
