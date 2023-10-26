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


def huggingface_download(repository: str, enable_progress_bar: bool = True) -> str:
    """
    Summary
    -------
    download the huggingface model

    Parameters
    ----------
    repository (str) : the name of the Hugging Face repository
    enable_progress_bar (bool?) : flag to disable the tqdm progress bar

    Returns
    -------
    repository_path (str) : local path to the downloaded repository
    """
    return snapshot_download(
        repository,
        resume_download=True,
        local_files_only=not has_internet_access(repository),
        tqdm_class=None if enable_progress_bar else DisableTqdm  # type: ignore
    )
