from typing import Generator

from fastapi import UploadFile
from fitz import Document as FitzDocument

from server.features.extraction.helpers import create_document
from server.features.extraction.models import Document
from server.features.extraction.models.document import Section


def extract_document_from_pdf(file_name: str, file_type: str, file: bytes) -> Document:
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

    return create_document(file_name, sections)


def extract_documents_from_pdf_requests(requests: list[UploadFile]) -> Generator[Document | None, None, None]:
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
            extract_document_from_pdf(*request.filename.rsplit('.', 1), file=request.file.read())
            if request.filename
            else None
        )
