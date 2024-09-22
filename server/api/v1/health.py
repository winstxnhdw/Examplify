from litestar import get


@get('/health', sync_to_thread=False)
def health() -> None:
    """
    Summary
    -------
    the `/health` will only return a 200 status code if the server is healthy
    """
    return
