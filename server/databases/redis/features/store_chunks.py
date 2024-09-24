from asyncio import gather
from typing import AsyncIterator, Callable, Iterable

from numpy import float32
from numpy.typing import NDArray

from server.databases.redis.wrapper import RedisAsync
from server.features.chunking.models import Chunk
from server.features.chunking.sentence_splitter import TextSplitter
from server.features.extraction.models import Document


async def store_chunks(
    redis: RedisAsync,
    chat_id: str,
    embedder: Callable[[str], NDArray[float32]],
    documents: Iterable[Document | None],
    chunker: Callable[[Document, TextSplitter], Iterable[Chunk]],
    text_splitter: TextSplitter,
) -> AsyncIterator[tuple[str, str] | tuple[None, None]]:
    """
    Summary
    -------
    store the vectorised chunks of text in Redis

    Parameters
    ----------
    pipeline (Pipeline) : the Redis pipeline to use
    chat_id (str): the chat ID
    embedder (Embedder) : the embedder to use
    documents (Sequence[Document]) : the documents to store
    chunker (Callable[[Document, TextSplitter], Sequence[Chunk]]) : the chunker to use
    text_splitter (TextSplitter) : the text splitter to use

    Yields
    ------
    document_identity (tuple[str, str] | tuple[None, None]) : the document identifier and semantic name
    """
    async with redis.pipeline() as pipeline:
        for document in documents:
            if not document:
                yield None, None
                continue

            yield document.id, document.semantic_identifier

            coroutines = [
                redis.hset(
                    chunk.source_id,
                    chunk.id,
                    mapping={
                        'vector': embedder(chunk.content).tobytes(),
                        'content': chunk.content,
                        'chat_id': chat_id,
                    },
                )
                for chunk in chunker(document, text_splitter)
            ]

            await gather(*coroutines)

        await pipeline.execute()
