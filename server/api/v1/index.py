from typing import Literal

from starlette.responses import PlainTextResponse

from server.api.v1 import v1


@v1.get('/', response_model=Literal['Welcome to v1 of the API!'])
def index() -> PlainTextResponse:
    """
    Summary
    -------
    the `/` route
    """
    return PlainTextResponse('Welcome to v1 of the API!')
