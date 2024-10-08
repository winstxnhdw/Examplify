from typing import Iterator

from fitz import Document as FitzDocument

from examplify.features.extraction.helpers import create_document
from examplify.features.extraction.models import Document
from examplify.features.extraction.models.document import Section
from examplify.features.extraction.types import File


def extract_document_from_pdf(file: File) -> Document:
    """
    Summary
    -------
    extract the text from a file

    Parameters
    ----------
    file (File): the file to extract the text from

    Returns
    -------
    document (Document): the parsed document
    """
    file_name, file_type = file['name'].rsplit('.', 1)
    file_data = file['data']

    with FitzDocument(stream=file_data, filetype=file_type) as document:
        sections = [
            Section(link=f'{file_name}#{page.number}', content=page.get_text(sort=True))  # type: ignore
            for page in document
        ]

    return create_document(file_data, file_name, sections)


def extract_documents_from_pdfs(files: list[File]) -> Iterator[Document]:
    """
    Summary
    -------
    extract the text from a list of requests

    Parameters
    ----------
    files (list[File]): the requests to extract the text from

    Yields
    ------
    documents (Document): the parsed document
    """
    return (extract_document_from_pdf(file) for file in files)
