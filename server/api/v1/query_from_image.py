from json import dumps
from typing import Annotated

from fastapi import Depends, UploadFile

from redis.asyncio import Redis

from server.api.v1 import v1
from server.config import Config
from server.databases.redis import create_query_parameters, redis_get
from server.databases.redis import redis_query as redis_query_helper
from server.dependencies import get_redis_client
from server.features import (
    LLM,
    Embedding,
    question_answering,
    extract_query_from_image_request
)
from server.features.llm.types import Message
from server.schemas.v1 import Answer


@v1.post('/{chat_id}/query_from_image')
async def query(
    redis: Annotated[Redis, Depends(get_redis_client)],
    chat_id: str,
    request: UploadFile,
    top_k: int = 5,
    store_query: bool = True
) -> Answer:
    """
    Summary
    -------
    the `/query` route provides an endpoint for performning retrieval-augmented generation
    """
    redis_query = redis_query_helper('tag', chat_id, top_k)

    query_str = extract_query_from_image_request(request)

    redis_query_parameters = create_query_parameters(
        Embedding().encode_query(query_str)
    )

    search_response = await redis.ft(Config.redis_index_name).search(
        redis_query,
        redis_query_parameters  # type: ignore  (this is a bug in the redis-py library)
    )

    context = ' '.join(
        document['content'] for document
        in search_response.docs # type: ignore
    )

    message_history: list[Message] = await redis_get(redis, f'chat:{chat_id}', _ := [])
    message_history.append({
        'role': 'user',
        'content': f'Given the following context:\n\n{context}\n\nPlease answer the following question:\n\n{query_str}'
    })

    messages = question_answering(message_history, LLM.query)

    if not store_query:
        await redis.set(f'chat:{chat_id}', dumps(messages))

    return Answer(messages=messages)
