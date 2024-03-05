from fastapi import UploadFile
from starlette.responses import PlainTextResponse

from server.api.debug import debug
from server.features import extract_text_from_image


@debug.post('/image_to_text')
def image_to_text(request: UploadFile) -> PlainTextResponse:
    """
    Summary
    -------
    the `/image_to_text` route provides an endpoint to extract text from an image
    """
    return PlainTextResponse(extract_text_from_image(request.file))
