from typing import Annotated

from fastapi import Depends, UploadFile
from redis.asyncio import Redis

from server.api.v1 import v1
from server.dependencies import get_redis_client
from server.features import extract_text_from_image, query_llm
from server.schemas.v1 import Answer


@v1.post('/{chat_id}/query_with_image')
async def query_with_image(
    redis: Annotated[Redis, Depends(get_redis_client)],
    chat_id: str,
    request: UploadFile,
    top_k: int = 5,
    store_query: bool = True
) -> Answer:
    """
    Summary
    -------
    the `/query_with_image` route is similar to `/query` but it accepts an image as input
    """
    extracted_query = extract_text_from_image(request.file)
    messages = await query_llm(redis, extracted_query, chat_id, top_k, store_query)

    return Answer(messages=messages)
