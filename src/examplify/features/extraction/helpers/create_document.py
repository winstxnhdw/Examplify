from hashlib import md5

from examplify.features.extraction.models import Document
from examplify.features.extraction.models.document import Section


def create_document(file_data: bytes, semantic_identifier: str, sections: list[Section]) -> Document:
    """
    Summary
    -------
    create a document

    Parameters
    ----------
    semantic_identifier (str): the semantic identifier of the document
    sections (list[Section]): the sections of the document

    Returns
    -------
    document (Document): the document
    """
    return Document(
        id=md5(file_data).hexdigest(),
        sections=sections,
        semantic_identifier=semantic_identifier,
    )
