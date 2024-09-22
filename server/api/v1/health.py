from litestar import get


@get('/health', sync_to_thread=False)
def health() -> None:
    """
    Summary
    -------
    the `/health` route
    """
    return
