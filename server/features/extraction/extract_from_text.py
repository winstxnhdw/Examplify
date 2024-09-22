from typing import BinaryIO, Iterator

from server.features.extraction.helpers import create_document
from server.features.extraction.models import Document
from server.features.extraction.models.document import Section
from server.features.extraction.types import File


def extract_document_from_text(file_name: str, file: BinaryIO) -> Document:
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
    return create_document(
        file_name,
        [
            Section(link=f'{file_name}#{i}', content=content.strip().replace('?', ''))
            for i, content in enumerate(file.read().decode('utf-8').split('\n\n'))
        ],
    )


def extract_documents_from_text_requests(files: list[File]) -> Iterator[Document]:
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
    return (extract_document_from_text(file['name'], file['data']) for file in files)
