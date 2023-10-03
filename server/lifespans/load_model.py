from asyncio import get_running_loop

from server.features import LLM


def load_language_model():
    """
    Summary
    -------
    download and load the language model
    """
    LLM.load()


async def load_model():
    """
    Summary
    -------
    download and load the model
    """
    await get_running_loop().run_in_executor(
        None,
        load_language_model
    )
