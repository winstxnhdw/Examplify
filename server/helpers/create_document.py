from uuid import uuid4

from server.features.extraction.models import Document
from server.features.extraction.models.document import Section


def create_document(semantic_identifier: str, sections: list[Section]) -> Document:
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
        id=str(uuid4()),
        sections=sections,
        semantic_identifier=semantic_identifier
    )
