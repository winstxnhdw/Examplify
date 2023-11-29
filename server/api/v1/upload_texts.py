from asyncio import gather
from typing import Annotated

from fastapi import Depends, UploadFile
from redis.asyncio import Redis
from starlette.exceptions import HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client
from server.features import (
    Embedding,
    extract_documents_from_text_requests,
    naive_chunk,
)
from server.schemas.v1 import DocumentSchema, Uploaded


@v1.post('/{chat_id}/upload_texts')
async def upload_texts(
    chat_id: str,
    requests: list[UploadFile],
    redis: Annotated[Redis, Depends(get_redis_client)]
):
    """
    Summary
    -------
    the `/upload_texts` route provides an endpoint for uploading texts to the server
    """
    embedder = Embedding()
    response: list[DocumentSchema | None] = [None] * len(requests)

    async with redis.pipeline() as pipeline:
        for i, document in enumerate(extract_documents_from_text_requests(requests)):
            if not document:
                raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'No file name!')

            response[i] = DocumentSchema(
                id=document.id,
                name=document.semantic_identifier
            )

            hashset_coroutines = [
                pipeline.hset(f'{Config.document_index_prefix}:{chunk.source_id}-{chunk.id}', mapping={
                    'vector': embedder.encode_normalise(chunk.content).tobytes(),
                    'content': chunk.content,
                    'tag': chat_id
                }) for chunk in naive_chunk(document)
            ]

            await gather(*hashset_coroutines) # type: ignore

        await pipeline.execute()

    return Uploaded(documents=response)  # type: ignore
