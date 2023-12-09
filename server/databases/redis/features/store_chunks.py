from asyncio import gather
from typing import (
    AsyncGenerator,
    Callable,
    Iterable,
    Protocol,
)

from numpy import float64
from numpy.typing import NDArray
from redis.asyncio.client import Pipeline

from server.config import Config
from server.features.chunking.models import Chunk
from server.features.chunking.sentence_splitter import TextSplitter
from server.features.extraction.models import Document


class Embedder(Protocol):
    """
    Summary
    -------
    a generic protocol for embedding text
    """
    def encode_normalise(self, sentences: str | list[str]) -> NDArray[float64]:
        """
        Summary
        -------
        encode a sentence or list of sentences into a normalised embedding
        """
        ...


async def store_chunks(
    pipeline: Pipeline,
    chat_id: str,
    embedder: Embedder,
    documents: Iterable[Document | None],
    chunker: Callable[[Document, TextSplitter], Iterable[Chunk]],
    text_splitter: TextSplitter
) -> AsyncGenerator[tuple[str, str] | tuple[None, None], None]:
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
    document_id (str) : the document ID
    semantic_identifier (str) : the document's semantic identifier
    """
    for document in documents:
        if not document:
            yield None, None
            continue

        yield document.id, document.semantic_identifier

        coroutines = [
            pipeline.hset(f'{Config.document_index_prefix}:{chunk.source_id}-{chunk.id}', mapping={
                'vector': embedder.encode_normalise(chunk.content).tobytes(),
                'content': chunk.content,
                'tag': chat_id
            }) for chunk in chunker(document, text_splitter)
        ]

        await gather(*coroutines)  # type: ignore

    await pipeline.execute()
