from typing import Annotated

from fastapi import Depends, UploadFile

from server.api.v1 import v1
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.features import Embedding, extract_text_from_image, query_llm
from server.schemas.v1 import Answer


@v1.post('/{chat_id}/query_with_image')
async def query_with_image(
    redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)],
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

    context = await redis.search(chat_id, embedding, top_k)
    message_history = await redis.get_messages(chat_id)
    messages = query_llm(extracted_query, context, message_history)

    if store_query:
        await redis.save_messages(chat_id, messages)

    return Answer(messages=messages)
