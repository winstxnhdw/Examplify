from typing import Annotated

from fastapi import Depends, UploadFile
from redis.asyncio import Redis

from server.api.v1 import v1
from server.databases.redis.features import save_messages, search
from server.databases.redis.helpers import redis_get
from server.dependencies import get_redis_client
from server.features import Embedding, extract_text_from_image, query_llm
from server.features.llm.types import Message
from server.schemas.v1 import Answer


@v1.post('/{chat_id}/query_with_image')
async def query_with_image(
    redis: Annotated[Redis, Depends(get_redis_client)],
    chat_id: str,
    request: UploadFile,
    top_k: int = 1,
    store_query: bool = True
) -> Answer:
    """
    Summary
    -------
    the `/query_with_image` route is similar to `/query` but it accepts an image as input
    """
    embedding = Embedding().encode_query(
        extracted_query := extract_text_from_image(request.file)
    )

    context = await search(redis, chat_id, embedding, top_k)
    message_history: list[Message] = await redis_get(redis, f'chat:{chat_id}', _ := [])
    messages =  query_llm(extracted_query, context, message_history)

    return Answer(messages = await save_messages(redis, chat_id, messages, store_query))
