from contextlib import asynccontextmanager
from typing import AsyncIterator

from litestar import Litestar

from examplify.utils import huggingface_download


@asynccontextmanager
async def download_embeddings(_: Litestar) -> AsyncIterator[None]:
    """
    Summary
    -------
    download the embeddings model

    Parameters
    ----------
    app (Litestar) : the Litestar application
    """
    huggingface_download('winstxnhdw/bge-base-en-v1.5-ct2')

    try:
        yield

    finally:
        pass
