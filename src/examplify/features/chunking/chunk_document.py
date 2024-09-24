from typing import Generator

from examplify.features.chunking.models import Chunk
from examplify.features.chunking.sentence_splitter import TextSplitter
from examplify.features.extraction.models import Document


def chunk_document(document: Document, text_splitter: TextSplitter) -> Generator[Chunk, None, None]:
    """
    Summary
    -------
    chunk a document into chunks of text

    Parameters
    ----------
    document (Document): the document to chunk

    Returns
    -------
    list[Chunk]: the chunks of text
    """
    return (
        Chunk(i, document.id, chunk)
        for i, section in enumerate(document.sections)
        for chunk in text_splitter.split_text(section.content)
        if len(chunk.split()) > 3
    )
