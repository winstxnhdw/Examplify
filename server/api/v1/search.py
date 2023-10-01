from typing import Annotated

from fastapi import Depends
from redis import Redis
from redis.commands.search.query import Query as RedisQuery

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client
from server.features import Embedding
from server.schemas.v1 import Search


@v1.post('/{chat_id}/search')
def search(
    chat_id: str,
    request: Search,
    redis: Annotated[Redis, Depends(get_redis_client)]
) -> str:
    """
    Summary
    -------
    the `/search` route provides an endpoint for searching the vector database
    """
    top_k = 1

    redis_query = (
        RedisQuery(f'(@tag:{{ {chat_id} }})=>[KNN {top_k} @vector $vec as score]')
            .sort_by('score')
            .return_fields("content", "score")
            .paging(0, top_k)
            .dialect(2)
    )

    redis_query_parameters: dict[str, str | int | float | bytes] = {
        'vec': Embedding().encode_normalise(request.query).tobytes()
    }

    return (
        redis
            .ft(Config.redis_index_name)
            .search(redis_query, redis_query_parameters)
            .docs[0]['content']  # type: ignore
    )
