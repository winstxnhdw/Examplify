from typing import Generator, Iterable

from ctranslate2 import Generator as LLMGenerator
from huggingface_hub import snapshot_download
from transformers.models.llama import LlamaTokenizerFast

from server.features.llm.types import Message


class LLM:
    """
    Summary
    -------
    a static class for generating text with an Large Language Model

    Methods
    -------
    stop_generation()
        stop the generation of text

    query(messages: Iterable[Message]) -> Message | None
        query the model

    generate(tokens_list: Iterable[list[str]]) -> Generator[str, None, None]
        generate text from a series/single prompt(s)
    """
    model_path = snapshot_download('winstxnhdw/Mistral-7B-Instruct-v0.1-ct2-int8')
    generator = LLMGenerator(model_path, device='cpu', compute_type='auto', inter_threads=1)
    tokeniser: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(model_path)
    stop_generator = False
    max_generation_length = 256
    max_prompt_length = 4096 - max_generation_length

    @classmethod
    def stop_generation(cls):
        """
        Summary
        -------
        stop the generation of text
        """
        cls.stop_generator = True


    @classmethod
    def query(cls, messages: Iterable[Message]) -> Message | None:
        """
        Summary
        -------
        query the model

        Parameters
        ----------
        messages (Iterable[Message]) : the messages to query the model with

        Returns
        -------
        answer (Message | None) : the answer to the query
        """
        prompts: str = cls.tokeniser.apply_chat_template(messages, tokenize=False)  # type: ignore
        tokens = cls.tokeniser(prompts).tokens()

        if len(tokens) <= cls.max_prompt_length:
            return None

        return {
            'role': 'assistant',
            'content': next(cls.generate([tokens]))
        }


    @classmethod
    def generate(cls, tokens_list: Iterable[list[str]]) -> Generator[str, None, None]:
        """
        Summary
        -------
        generate text from a series/single prompt(s)

        Parameters
        ----------
        prompt (str) : the prompt to generate text from

        Yields
        -------
        answer (str) : the generated answer
        """
        cls.stop_generator = False

        yield from (
            cls.tokeniser.decode(result.sequences_ids[0]) for result in cls.generator.generate_iterable(
                tokens_list,
                repetition_penalty=1.2,
                max_length=cls.max_generation_length,
                include_prompt_in_result=False,
                sampling_topp=0.9,
                sampling_temperature=0.9,
                callback=lambda _: cls.stop_generator
            )
        )
