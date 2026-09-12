from huggingface_hub import snapshot_download

from src.utils.network.has_internet_access import has_internet_access


def huggingface_download(repository: str) -> str:
    return snapshot_download(
        repository,
        local_files_only=not has_internet_access(repository),
    )
