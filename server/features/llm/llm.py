from typing import Generator, Sequence

from ctranslate2 import Generator as LLMGenerator
from transformers.models.llama import LlamaTokenizerFast

from server.config import Config
from server.features.llm.types import Message
from server.helpers import huggingface_download


class LLM:
    """
    Summary
    -------
    a static class for generating text with an Large Language Model

    Methods
    -------
    set_static_prompt(static_user_prompt: str, static_assistant_prompt: str) -> int
        set the model's static prompt

    load()
        download and load the language model

    query(messages: Sequence[Message]) -> Message | None
        query the model

    generate(tokens_list: Sequence[list[str]]) -> Generator[str, None, None]
        generate text from a series/single prompt(s)
    """
    generator: LLMGenerator
    tokeniser: LlamaTokenizerFast
    max_generation_length: int
    max_prompt_length: int
    static_prompt: list[str]


    @classmethod
    def set_static_prompt(cls, static_user_prompt: str, static_assistant_prompt: str) -> int:
        """
        Summary
        -------
        set the model's static prompt

        Parameters
        ----------
        static_user_prompt (str) : the static user prompt
        static_assistant_prompt (str) : the static assistant prompt

        Returns
        -------
        tokens (int) : the number of tokens in the static prompt
        """
        static_prompts: list[Message] = [{
            'role': 'user',
            'content': static_user_prompt
        },
        {
            'role': 'assistant',
            'content': static_assistant_prompt
        }]

        system_prompt = cls.tokeniser.apply_chat_template(
            static_prompts,
            add_generation_prompt=True,
            tokenize=False
        )

        cls.static_prompt = cls.tokeniser(system_prompt).tokens()

        return len(cls.static_prompt)


    @classmethod
    def load(cls):
        """
        Summary
        -------
        download and load the language model
        """
        model_path = huggingface_download('winstxnhdw/openchat-3.5-ct2-int8')
        device = 'cuda' if Config.use_cuda else 'cpu'

        cls.generator = LLMGenerator(model_path, device=device, compute_type='auto', inter_threads=1)
        cls.tokeniser = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)
        cls.max_generation_length = 1024
        cls.max_prompt_length = 4096 - cls.max_generation_length - cls.set_static_prompt(
            'You may be given the following chat history. '
            'Answer the question based on the context (if provided) as truthfully as you are able to. '
            'If you do not know the answer, you may respond with "I do not know". '
            'What is the capital of Japan?',
            'Tokyo.'
        )


    @classmethod
    def query(cls, messages: Sequence[Message]) -> Message | None:
        """
        Summary
        -------
        query the model

        Parameters
        ----------
        messages (Sequence[Message]) : the messages to query the model with

        Returns
        -------
        answer (Message | None) : the answer to the query
        """
        prompts: str = cls.tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = cls.tokeniser(prompts).tokens()

        if len(tokens) > cls.max_prompt_length:
            return None

        return {
            'role': 'assistant',
            'content': next(cls.generate([tokens]))
        }


    @classmethod
    def generate(cls, tokens_list: Sequence[list[str]]) -> Generator[str, None, None]:
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
        return (
            cls.tokeniser.decode(result.sequences_ids[0]) for result in cls.generator.generate_iterable(
                tokens_list,
                repetition_penalty=1.2,
                max_length=cls.max_generation_length,
                static_prompt=cls.static_prompt,
                include_prompt_in_result=False,
                sampling_topp=0.9,
                sampling_temperature=0.9
            )
        )
