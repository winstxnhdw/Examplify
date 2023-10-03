from asyncio import get_running_loop

from ctranslate2 import Generator
from transformers.models.llama import LlamaTokenizerFast

from server.features import LLM
from server.helpers import huggingface_download


def load_language_model():
    """
    Summary
    -------
    download and load the language model
    """
    model_path = huggingface_download('winstxnhdw/Mistral-7B-Instruct-v0.1-ct2-int8')
    LLM.generator = Generator(model_path, device='cpu', compute_type='auto', inter_threads=1)
    LLM.tokeniser = LlamaTokenizerFast.from_pretrained(model_path)


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
