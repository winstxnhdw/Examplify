from contextlib import asynccontextmanager
from typing import AsyncIterator

from litestar import Litestar

from server.features import LLM


@asynccontextmanager
async def chat_model(app: Litestar) -> AsyncIterator[None]:
    """
    Summary
    -------
    load the chat model

    Parameters
    ----------
    app (Litestar) : the Litestar application
    """
    app.state.chat = LLM.load()

    try:
        yield
    finally:
        pass
