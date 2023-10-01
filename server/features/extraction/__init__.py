from uuid import uuid4

from fitz import Document as FitzDocument

from server.features.extraction.models import Document
from server.features.extraction.models.document import Section


def extract_text(file_name: str, file: bytes, file_type: str) -> Document:
    """
    Summary
    -------
    extract the text from a file

    Parameters
    ----------
    file_name (str): the name of the file
    file (bytes): the file
    file_type (str): the type of the file

    Returns
    -------
    document (Document): the parsed document
    """
    with FitzDocument(stream=file, filetype=file_type) as document:
        sections = [
            Section(link=f'{file_name}#{page.number}', content=page.get_text(sort=True))  # type: ignore
            for page in document
        ]

    return Document(
        id=str(uuid4()),
        sections=sections,
        semantic_identifier=file_name
    )
