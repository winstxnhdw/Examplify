from contextlib import asynccontextmanager
from typing import AsyncIterator

from litestar import Litestar
from nltk.data import find, path
from nltk.downloader import download


def find_or_download_nltk(download_directory: str):
    """
    Summary
    -------
    a function to find or download NLTK

    Parameters
    ----------
    download_directory (str) : the directory at which NLTK should be installed or found
    """
    try:
        find('tokenizers/punkt')

    except LookupError:
        download('punkt', download_dir=download_directory)


@asynccontextmanager
async def download_nltk(app: Litestar) -> AsyncIterator[None]:
    """
    Summary
    -------
    an async function to download NLTK

    Parameters
    ----------
    app (Litestar) : the Litestar application
    """
    download_directory = '/home/user/.cache/nltk'
    path.append(download_directory)
    find_or_download_nltk(download_directory)

    try:
        yield
    finally:
        pass
