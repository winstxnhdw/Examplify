from typing import Annotated

from fastapi import Depends, UploadFile
from llama_index.text_splitter import SentenceSplitter
from redis import Redis
from starlette.exceptions import HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from server.api.v1 import v1
from server.config import Config
from server.dependencies import get_redis_client
from server.features import LLM, Embedding, chunk_document, extract_text
from server.features.extraction import Document


@v1.post('/{chat_id}/upload_files')
def upload_files(
    chat_id: str,
    requests: list[UploadFile],
    redis: Annotated[Redis, Depends(get_redis_client)]
):
    """
    Summary
    -------
    the `/upload_files` route provides an endpoint for uploading a file to the server
    """
    documents: list[Document] = []
    embedder = Embedding()

    for request in requests:
        if not request.filename:
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'No file name!')

        file_name, file_type = request.filename.rsplit('.', 1)
        documents.append(extract_text(file_name, request.file.read(), file_type))

    with redis.pipeline() as pipeline:
        for document in documents:
            chunks = chunk_document(document, SentenceSplitter(
                chunk_size=128,
                chunk_overlap=0,
                tokenizer=LLM.tokeniser
            ))

            for chunk in chunks:
                pipeline.hset(f'{Config.document_index_prefix}:{chunk.source_id}-{chunk.id}', mapping={
                    'vector': embedder.encode_normalise(chunk.content).tobytes(),
                    'content': chunk.content,
                    'tag': chat_id
                })

        pipeline.execute()

    return [document.id for document in documents]
