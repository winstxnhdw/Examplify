from asyncio import get_running_loop

from server.helpers.network import huggingface_download


async def download_embeddings():
    """
    Summary
    -------
    download the embeddings model
    """
    await get_running_loop().run_in_executor(
        None,
        lambda: huggingface_download('winstxnhdw/bge-base-en-v1.5-ct2', False)
    )
