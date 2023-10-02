from asyncio import gather
from typing import Annotated

from fastapi import Depends, UploadFile
from redis.asyncio import Redis
from starlette.exceptions import HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client
from server.features import LLM, Embedding, chunk_document
from server.features.chunking.sentence_splitter import SentenceSplitter
from server.features.extraction import extract_texts_from_requests


@v1.post('/{chat_id}/upload_files')
async def upload_files(
    chat_id: str,
    requests: list[UploadFile],
    redis: Annotated[Redis, Depends(get_redis_client)]
):
    """
    Summary
    -------
    the `/upload_files` route provides an endpoint for uploading a file to the server
    """
    embedder = Embedding()
    text_splitter = SentenceSplitter(LLM.tokeniser, chunk_size=128, chunk_overlap=0)

    async with redis.pipeline() as pipeline:
        for document in extract_texts_from_requests(requests):
            if not document:
                raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'No file name!')

            for chunk in chunk_document(document, text_splitter):
                pipeline.hset(f'{Config.document_index_prefix}:{chunk.source_id}-{chunk.id}', mapping={
                    'vector': embedder.encode_normalise(chunk.content).tobytes(),
                    'content': chunk.content,
                    'tag': chat_id
                })

        await pipeline.execute()

    # return [document.id for document in documents]
