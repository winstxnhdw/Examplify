from typing import Generator

from nltk.tokenize import sent_tokenize

from server.features.chunking.models import Chunk
from server.features.extraction.models import Document


def naive_chunk(document: Document) -> Generator[Chunk, None, None]:
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
        for chunk in sent_tokenize(section.content)
        if len(chunk.split()) > 3
    )
