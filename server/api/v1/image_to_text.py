from fastapi import UploadFile

from server.api.v1 import v1
from server.features import extract_text_from_image


@v1.post('/image_to_text')
def image_to_text(request: UploadFile) -> str:
    """
    Summary
    -------
    the `/image_to_text` route provides an endpoint to extract text from an image
    """
    return extract_text_from_image(request.file)
