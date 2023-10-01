from typing import Annotated

from fastapi import Depends
from redis import Redis
from redis.commands.search.query import Query as RedisQuery

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client
from server.features import Embedding, question_answering
from server.schemas.v1 import Answer, Query


@v1.post('/{chat_id}/query')
def query(
    chat_id: str,
    request: Query,
    redis: Annotated[Redis, Depends(get_redis_client)]
) -> Answer:
    """
    Summary
    -------
    the `/query` route provides an endpoint for performning retrieval-augmented generation
    """
    redis_query = (
        RedisQuery(f'(@tag:{{ {chat_id} }})=>[KNN 5 @vector $vec as score]')
            .sort_by('score')
            .return_fields("content", "score")
            .paging(0, 2)
            .dialect(2)
    )

    redis_query_parameters: dict[str, str | int | float | bytes] = {
        'vec': Embedding().encode_normalise(request.query).tobytes()
    }

    context = '\n'.join(
        document['content'] for document in
        redis
            .ft(Config.redis_index_name)
            .search(redis_query, redis_query_parameters)
            .docs  # type: ignore
    )

    return Answer(messages=question_answering(
        request.query,
        context,
        request.messages
    ))
