from typing import Generator
from uuid import uuid4

from fastapi import UploadFile

from tesserocr import PyTessBaseAPI
from io import BytesIO
from PIL import Image

from server.features.extraction.models import Document
from server.features.extraction.models.document import Section

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