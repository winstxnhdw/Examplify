from http.client import HTTPConnection
from typing import Any

from huggingface_hub import snapshot_download
from tqdm.asyncio import tqdm_asyncio


class DisableTqdm(tqdm_asyncio):  # type: ignore
    """
    Summary
    -------
    a class that disables the tqdm progress bar
    """
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs['disable'] = True
        super().__init__(*args, **kwargs)


def has_internet_access() -> bool:
    """
    Summary
    -------
    check if the server has internet access
    """
    connection = HTTPConnection('1.1.1.1', timeout=1)

    try:
        connection.request('HEAD', '/')
        return True

    except (TimeoutError, OSError):
        return False

    finally:
        connection.close()


def huggingface_download(repository: str, enable_progress_bar: bool = True) -> str:
    """
    Summary
    -------
    download the huggingface model
    """
    return snapshot_download(
        repository,
        resume_download=True,
        local_files_only=not has_internet_access(),
        tqdm_class=None if enable_progress_bar else DisableTqdm  # type: ignore
    )
