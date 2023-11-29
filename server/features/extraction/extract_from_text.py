from typing import Generator

from fastapi import UploadFile

from server.features.extraction.helpers import create_document
from server.features.extraction.models import Document
from server.features.extraction.models.document import Section


def extract_document_from_text(file_name: str, file: bytes) -> Document:
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
    return create_document(file_name, [
        Section(link=f'{file_name}#{i}', content=content.strip())
        for i, content
        in enumerate(file.decode('utf-8').split('\n\n'))
    ])


def extract_documents_from_text_requests(requests: list[UploadFile]) -> Generator[Document | None, None, None]:
    """
    Summary
    -------
    extract the text from a list of requests

    Parameters
    ----------
    requests (list[UploadFile]): the requests to extract the text from

    Yields
    ------
    documents (Document): the parsed document
    """
    for request in requests:
        yield (
            extract_document_from_text(request.filename.rsplit('.', 1)[0], file=request.file.read())
            if request.filename
            else None
        )
