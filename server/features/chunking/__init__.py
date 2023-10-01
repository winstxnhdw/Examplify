from llama_index.text_splitter.types import MetadataAwareTextSplitter

from server.features.chunking.models import Chunk
from server.features.extraction.models import Document


def chunk_document(document: Document, text_splitter: MetadataAwareTextSplitter) -> list[Chunk]:
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
    return [
        Chunk(chunk)
        for section in document.sections
        for chunk in text_splitter.split_text(section.content)
    ]
