from typing import Annotated

from fastapi import Depends, UploadFile
from starlette.exceptions import HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from server.api.v1 import v1
from server.databases.redis.features import store_chunks
from server.databases.redis.wrapper import RedisAsyncWrapper
from server.dependencies import get_redis_client
from server.features import (
    LLM,
    Embedding,
    chunk_document,
    extract_documents_from_pdf_requests,
)
from server.features.chunking import SentenceSplitter
from server.schemas.v1 import DocumentSchema, Uploaded


@v1.put('/{chat_id}/upload_pdfs')
async def upload_pdfs(
    chat_id: str,
    requests: list[UploadFile],
    redis: Annotated[RedisAsyncWrapper, Depends(get_redis_client)]
) -> Uploaded:
    """
    Summary
    -------
    the `/upload_pdfs` route provides an endpoint for uploading PDFs to the server
    """
    embedder = Embedding()
    text_splitter = SentenceSplitter(LLM.tokeniser, chunk_size=128, chunk_overlap=0)
    responses = []
    chunk_generator = store_chunks(
        redis,
        chat_id,
        embedder,
        extract_documents_from_pdf_requests(requests),
        chunk_document,
        text_splitter
    )

    async for file_id, file_name in chunk_generator:
        if not file_name or not file_id:
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Invalid file name!')

        responses.append(DocumentSchema(id=file_id, name=file_name))

    return Uploaded(documents=responses)
