from http.client import HTTPConnection


def has_internet_access(repository: str) -> bool:
    connection = HTTPConnection("huggingface.co", timeout=1)

    try:
        connection.request("HEAD", f"/{repository}")

    except (TimeoutError, OSError):
        return False

    else:
        return True

    finally:
        connection.close()
