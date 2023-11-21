from typing import Generator
from uuid import uuid4

from fastapi import UploadFile
from fitz import Document as FitzDocument

from tesserocr import PyTessBaseAPI
from io import BytesIO
from PIL import Image

from server.features.extraction.models import Document
from server.features.extraction.models.document import Section


def extract_text(file_name: str, file_type: str, file: bytes) -> Document:
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

def extract_texts_from_image(file_name: str, image: Image) -> Document:
    """
    Summary
    -------
    extract the text from an image

    Parameters
    ----------
    file_name (str): the name of the file
    file (bytes): the file

    Returns
    -------
    document (Document): the parsed document
    """
    with PyTessBaseAPI(path='/usr/share/tesseract-ocr/5/tessdata') as ocr:
        ocr.SetImage(image)
        sections = [ocr.GetUTF8Text()]

    return Document(
        id=str(uuid4()),
        sections=sections,
        semantic_identifier=file_name
    )

def extract_texts_from_requests(requests: list[UploadFile]) -> Generator[Document | None, None, None]:
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
            extract_text(*request.filename.rsplit('.', 1), file=request.file.read())
            if request.filename
            else None
        )

def extract_texts_from_image_requests(requests: list[UploadFile]) -> Generator[Document | None, None, None]:
    """
    Summary
    -------
    extract the text from a list of requests containing images

    Parameters
    ----------
    requests (list[UploadFile]): the requests to extract the text from

    Yields
    ------
    documents (Document): the parsed document
    """
    for request in requests:
        with Image.open(BytesIO(request.file.read())) as img:
            yield (
                extract_texts_from_image(request.filename.rsplit('.', 1)[0], image=img)
                if request.filename
                else None
            )