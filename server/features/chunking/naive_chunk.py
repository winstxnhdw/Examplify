from typing import Generator

from nltk.tokenize import word_tokenize

from server.features.chunking.models import Chunk
from server.features.extraction.models import Document


def naive_chunk(document: Document) -> Generator[Chunk, None, None]:
    """
    Summary
    -------
    very naive document chunker

    Parameters
    ----------
    document (Document): the document to chunk

    Returns
    -------
    list[Chunk]: the chunks of text
    """
    sentences: list[str] = []
    sentence: list[str] = []

    for section in document.sections:
        for word in word_tokenize(section.content, preserve_line=True):
            if word in '.!' or word[0].isupper():
                sentences.append(' '.join(sentence))
                sentence.clear()

            sentence.append(word)

    return (
        Chunk(i, document.id, chunk)
        for i, chunk in enumerate(sentences)
        if len(chunk.split()) > 3
    )
