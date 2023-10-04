from typing import Generator, Iterable

from ctranslate2 import Generator as LLMGenerator
from transformers.models.llama import LlamaTokenizerFast

from server.features.llm.types import Message
from server.helpers import huggingface_download


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
    generator: LLMGenerator
    tokeniser: LlamaTokenizerFast
    max_generation_length: int
    max_prompt_length: int
    static_prompt: list[str]
    stop_generator = False

    @classmethod
    def load(cls):
        """
        Summary
        -------
        download and load the language model
        """
        model_path = huggingface_download('winstxnhdw/Mistral-7B-Instruct-v0.1-ct2-int8')
        cls.generator = LLMGenerator(model_path, device='cpu', compute_type='auto', inter_threads=1)
        cls.tokeniser = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)

        system_prompt = cls.tokeniser.apply_chat_template((
            {
                'content': 'You are a helpful AI assistant. You are given the following chat history. Answer the question based on the context provided as truthfully as you are able to. If you do not know the answer, you may respond with "I do not know". What is the Baloney Detection Kit?',
                'role': 'user'
            },
            {
                'content': 'The Baloney Detection Kit is a a set of cognitive tools and techniques that fortify the mind against penetration by falsehoods. It was created by Carl Sagan.',
                'role': 'assistant'
            }
        ), tokenize=False)

        cls.static_prompt = cls.tokeniser(system_prompt).tokens()
        cls.max_generation_length = 512
        cls.max_prompt_length = 4096 - cls.max_generation_length - len(cls.static_prompt)


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
        prompts: str = cls.tokeniser.apply_chat_template(messages, tokenize=False)
        tokens = cls.tokeniser(prompts).tokens()

        if len(tokens) > cls.max_prompt_length:
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

        return (
            cls.tokeniser.decode(result.sequences_ids[0]) for result in cls.generator.generate_iterable(
                tokens_list,
                repetition_penalty=1.2,
                max_length=cls.max_generation_length,
                static_prompt=cls.static_prompt,
                include_prompt_in_result=False,
                sampling_topp=0.9,
                sampling_temperature=0.9,
                callback=lambda _: cls.stop_generator
            )
        )
