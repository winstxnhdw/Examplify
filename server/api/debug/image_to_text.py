from fastapi import UploadFile

from server.api.debug import debug
from server.features import extract_text_from_image


@debug.post('/image_to_text')
def image_to_text(request: UploadFile) -> str:
    """
    Summary
    -------
    the `/image_to_text` route provides an endpoint to extract text from an image
    """
    return extract_text_from_image(request.file)
