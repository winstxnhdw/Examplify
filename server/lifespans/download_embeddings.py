from asyncio import get_running_loop

from huggingface_hub import snapshot_download


async def download_embeddings():
    """
    Summary
    -------
    download the embeddings model
    """
    get_running_loop().run_in_executor(None, lambda: snapshot_download('winstxnhdw/bge-base-en-v1.5-ct2'))
