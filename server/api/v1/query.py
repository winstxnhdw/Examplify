from typing import Annotated

from fastapi import Depends

from server.api.v1 import v1
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.features import LLM, Embedding, question_answering
from server.schemas.v1 import Answer, Query


@v1.post('/{chat_id}/query')
async def query(
    redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)],
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
    context = await redis.search(chat_id, embedding, top_k)
    message_history = await redis.get_messages(chat_id)
    messages = await question_answering(request.query, context, message_history, LLM.query)

    if store_query:
        await redis.save_messages(chat_id, messages)

    return Answer(messages=messages)
