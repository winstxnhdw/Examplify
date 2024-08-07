from typing import BinaryIO, Generator

from fastapi import UploadFile
from PIL.Image import open as open_image
from tesserocr import image_to_text

from server.features.extraction.helpers import create_document
from server.features.extraction.models import Document
from server.features.extraction.models.document import Section


def extract_text_from_image(file: BinaryIO) -> str:
    """
    Summary
    -------
    extract the text from an image

    Parameters
    ----------
    file (BinaryIO): the image

    Returns
    -------
    text (str): the text in the image
    """
    with open_image(file) as image:
        return image_to_text(image)


def extract_document_from_image(file_name: str, file: BinaryIO) -> Document:
    """
    Summary
    -------
    extract a document from an image

    Parameters
    ----------
    file_name (str): the name of the file
    file (BinaryIO): the image

    Returns
    -------
    document (Document): the parsed document
    """
    return create_document(file_name, [Section(link=f'{file_name}', content=extract_text_from_image(file))])


def extract_documents_from_image_requests(requests: list[UploadFile]) -> Generator[Document | None, None, None]:
    """
    Summary
    -------
    extract documents from a list of requests containing images

    Parameters
    ----------
    requests (list[UploadFile]): the requests to extract the text from

    Yields
    ------
    documents (Document): the parsed document
    """
    for request in requests:
        yield (
            extract_document_from_image(request.filename.rsplit('.', 1)[0], request.file) if request.filename else None
        )
