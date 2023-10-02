from asyncio import get_event_loop

from nltk import download
from nltk.data import find, path


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


async def download_nltk():
    """
    Summary
    -------
    an async function to download NLTK
    """
    download_directory = '/home/user/.cache/nltk'
    path.append(download_directory)

    await get_event_loop().run_in_executor(
        None,
        lambda: find_or_download_nltk(download_directory)
    )
