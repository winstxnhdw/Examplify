from typing import Iterator

from ctranslate2 import Generator
from transformers.models.llama import LlamaTokenizerFast

from examplify.config import Config
from examplify.features.chat.types import Message
from examplify.utils import huggingface_download


class ChatModel:
    """
    Summary
    -------
    a class for generating text with an Large Language Model

    Methods
    -------
    set_static_prompt(static_user_prompt: str, static_assistant_prompt: str) -> int
        set the model's static prompt

    query(messages: Sequence[Message]) -> Message | None
        query the model

    generate(tokens_list: Sequence[list[str]]) -> Awaitable[str]
        generate text from a series/single prompt(s)
    """

    __slots__ = (
        'generator',
        'tokeniser',
        'min_query_length',
        'max_context_length',
        'max_generation_length',
        'max_query_length',
        'static_prompt',
    )

    def __init__(
        self,
        generator: Generator,
        tokeniser: LlamaTokenizerFast,
        min_query_length: int,
        max_context_length: int,
        max_generation_length: int,
    ):
        self.max_query_length = max_context_length - max_generation_length

        if self.max_query_length < min_query_length:
            raise ValueError('The minimum query length cannot be greater than the maximum query length!')

        self.generator = generator
        self.tokeniser = tokeniser
        self.min_query_length = min_query_length
        self.max_context_length = max_context_length
        self.max_generation_length = max_generation_length
        self.static_prompt = []

    def __len__(self) -> int:
        return len(self.static_prompt)

    def set_static_prompt(self, static_user_prompt: str, static_assistant_prompt: str) -> bool:
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
        static_prompts: list[Message] = [
            {
                'role': 'user',
                'content': static_user_prompt,
            },
            {
                'role': 'assistant',
                'content': static_assistant_prompt,
            },
        ]

        system_prompt = self.tokeniser.apply_chat_template(static_prompts, add_generation_prompt=True, tokenize=False)
        static_prompt = self.tokeniser(system_prompt).tokens()
        max_query_length = self.max_context_length - self.max_generation_length - len(self.static_prompt)

        if max_query_length < self.min_query_length:
            return False

        self.static_prompt = static_prompt
        self.max_query_length = max_query_length

        return True

    def query(self, messages: list[Message]) -> Iterator[str] | None:
        """
        Summary
        -------
        query the model

        Parameters
        ----------
        messages (list[Message]) : the messages to query the model with

        Returns
        -------
        answer (Message | None) : the answer to the query
        """
        prompts: str = self.tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokeniser(prompts).tokens()

        if len(tokens) > self.max_query_length:
            return None

        return self.generate(tokens)

    def generate(self, tokens: list[str]) -> Iterator[str]:
        """
        Summary
        -------
        generate text from a series/single prompt(s)

        Parameters
        ----------
        tokens (list[str]) : the tokens to generate text from

        Yields
        -------
        answer (str) : the generated answer
        """
        for result in self.generator.generate_tokens(
            tokens,
            repetition_penalty=1.2,
            max_length=self.max_generation_length,
            static_prompt=self.static_prompt,
            sampling_topp=0.9,
            sampling_temperature=0.9,
        ):
            if result.is_last:
                return

            yield self.tokeniser.decode(result.token_id)


def get_chat_model() -> ChatModel:
    """
    Summary
    -------
    download and load the language model

    Returns
    -------
    model (Chat) : the language model
    """
    model_path = huggingface_download('winstxnhdw/openchat-3.6-ct2-int8')
    tokeniser = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)
    generator = Generator(
        model_path,
        'cuda' if Config.use_cuda else 'cpu',
        compute_type='auto',
        inter_threads=Config.chat_model_threads,
        max_queued_batches=-1,
    )

    min_query_length = 64
    max_context_length = 4096
    max_generation_length = 1024

    return ChatModel(generator, tokeniser, min_query_length, max_context_length, max_generation_length)
