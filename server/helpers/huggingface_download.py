from http.client import HTTPConnection

from huggingface_hub import snapshot_download


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

    except ConnectionError:
        return False

    finally:
        connection.close()


def huggingface_download(repository: str) -> str:
    """
    Summary
    -------
    download the huggingface model
    """
    return snapshot_download(repository, resume_download=True, local_files_only=not has_internet_access())
