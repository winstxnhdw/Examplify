from io import BytesIO
from typing import Annotated

from litestar import post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import ServerSentEvent
from litestar.status_codes import HTTP_200_OK

from server.features.extraction import file_to_text


@post('/files/text', status_code=HTTP_200_OK)
async def files_to_text(
    data: Annotated[list[UploadFile], Body(media_type=RequestEncodingType.MULTI_PART)],
) -> ServerSentEvent:
    """
    Summary
    -------
    an endpoint to extract text from files
    """

    return ServerSentEvent(file_to_text(BytesIO(await file.read()), file.filename) for file in data)
