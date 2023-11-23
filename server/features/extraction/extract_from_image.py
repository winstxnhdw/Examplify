from io import BytesIO
from typing import Generator
from uuid import uuid4

from fastapi import UploadFile
from PIL.Image import Image
from PIL.Image import open as open_image
from tesserocr import PyTessBaseAPI

from server.features.extraction.models import Document
from server.features.extraction.models.document import Section


def extract_document_from_image(file_name: str, image: Image) -> Document:
    """
    Summary
    -------
    extract a document from an image

    Parameters
    ----------
    file_name (str): the name of the file
    file (bytes): the file

    Returns
    -------
    document (Document): the parsed document
    """
    section = Section(
        link=f'{file_name}',
        content=extract_text_from_image(image)
    )

    return Document(
        id=str(uuid4()),
        sections=[section],
        semantic_identifier=file_name
    )


def extract_text_from_image(image: Image) -> str:
    """
    Summary
    -------
    extract the text from an image

    Parameters
    ----------
    file (bytes): the file

    Returns
    -------
    text (str): the text in the image
    """
    with PyTessBaseAPI() as ocr:
        ocr.SetImage(image)
        return ocr.GetUTF8Text()


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
        with open_image(BytesIO(request.file.read())) as image:
            yield (
                extract_document_from_image(request.filename.rsplit('.', 1)[0], image=image)
                if request.filename
                else None
            )


def extract_query_from_image_request(request: UploadFile) -> str:
    """
    Summary
    -------
    extract a text for a request containing an image

    Parameters
    ----------
    request (UploadFile): the request to extract the text from

    Returns
    ------
    text (str): the parsed text
    """
    with open_image(BytesIO(request.file.read())) as image:
        return extract_text_from_image(image)
