from asyncio import AbstractEventLoop, Queue, get_running_loop
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Self
from weakref import finalize

from ctranslate2 import Generator
from transformers.models.qwen2 import Qwen2TokenizerFast

from src.features.chat.prompts import Prompt
from src.features.chat.protocol import ChatAgentProtocol
from src.utils import huggingface_download


class QueryLengthError(Exception):
    def __init__(self) -> None:
        super().__init__("The minimum query length cannot be greater than the maximum query length!")


class ChatModel(ChatAgentProtocol):
    __slots__ = (
        "generator",
        "max_context_length",
        "max_generation_length",
        "max_query_length",
        "min_query_length",
        "static_prompt",
        "tokeniser",
    )

    def __init__(
        self,
        generator: Generator,
        tokeniser: Qwen2TokenizerFast,
        min_query_length: int,
        max_context_length: int,
        max_generation_length: int,
        chat_model_threads: int,
    ) -> None:
        self.max_query_length = max_context_length - max_generation_length

        if self.max_query_length < min_query_length:
            raise QueryLengthError

        self.generator = generator
        self.tokeniser = tokeniser
        self.min_query_length = min_query_length
        self.max_context_length = max_context_length
        self.max_generation_length = max_generation_length
        self.static_prompt = []
        self.thread_pool = ThreadPoolExecutor(chat_model_threads)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_) -> None:
        del self.generator
        del self.tokeniser

    def encode_messages(self, prompts: Iterable[Prompt]) -> list[str]:
        prompts_dict = [{"role": prompt.role, "content": prompt.content} for prompt in prompts]
        prompt = self.tokeniser.apply_chat_template(prompts_dict, add_generation_prompt=True, tokenize=False)

        return self.tokeniser(prompt)._encodings[0].tokens  # pyright: ignore [reportOptionalSubscript. reportAssignmentType]  # noqa: SLF001

    def set_static_prompt(self, static_user_prompt: str, static_assistant_prompt: str) -> bool:
        static_prompts: list[Prompt] = [
            Prompt(role="user", content=static_user_prompt),
            Prompt(role="assistant", content=static_assistant_prompt),
        ]

        static_prompt = self.encode_messages(static_prompts)
        max_query_length = self.max_context_length - self.max_generation_length - len(static_prompt)

        if max_query_length < self.min_query_length:
            return False

        self.static_prompt = static_prompt
        self.max_query_length = max_query_length

        return True

    def generate_from_another_thread(
        self,
        queue: Queue[str | None],
        tokens: list[str],
        cancel_event: Event,
        loop: AbstractEventLoop,
    ) -> None:
        generator = self.generator.generate_tokens(
            tokens,
            repetition_penalty=1.2,
            max_length=self.max_generation_length,
            static_prompt=self.static_prompt,
            sampling_topp=0.9,
            sampling_temperature=0.9,
        )

        for result in generator:
            if cancel_event.is_set():
                print("generation cancelled!!!!")
                generator.close()
                break

            if result.is_last:
                break

            loop.call_soon_threadsafe(
                queue.put_nowait,
                self.tokeniser.backend_tokenizer.decoder.decode((result.token,)),
            )

        loop.call_soon_threadsafe(queue.put_nowait, None)

    async def generate_stream(self, tokens: list[str]) -> AsyncIterator[str]:
        queue: Queue[str | None] = Queue()
        cancel_event = Event()
        loop = get_running_loop()
        self.thread_pool.submit(self.generate_from_another_thread, queue, tokens, cancel_event, loop)
        finalize(loop, cancel_event.set)

        while True:
            token = await queue.get()

            if token is None:
                break

            yield token

    # async def generate(self, prompts: Iterable[Prompt]) -> AsyncIterator[str] | None:
    #     if len(tokens := self.encode_messages(prompts)) > self.max_query_length:
    #         return None

    #     return self.generate_stream(tokens)

    def generate(
        self,
        prompts: Iterable[Prompt],
    ) -> None:
        if len(tokens := self.encode_messages(prompts)) > self.max_query_length:
            return None

        generator = self.generator.generate_tokens(
            tokens,
            repetition_penalty=1.2,
            max_length=self.max_generation_length,
            static_prompt=self.static_prompt,
            sampling_topp=0.9,
            sampling_temperature=0.9,
        )

        return (
            self.tokeniser.backend_tokenizer.decoder.decode([result.token])
            for result in generator
            if not result.is_last
        )


def get_ctranslate_model(chat_model_threads: int, *, use_cuda: bool) -> ChatModel:
    model_path = huggingface_download("winstxnhdw/Qwen2.5-7B-Instruct-ct2-int8")
    tokeniser = Qwen2TokenizerFast.from_pretrained(model_path, local_files_only=True, legacy=False)
    generator = Generator(
        model_path,
        "cuda" if use_cuda else "cpu",
        compute_type="auto",
        inter_threads=chat_model_threads,
        max_queued_batches=-1,
    )

    min_query_length = 64
    max_context_length = 131072
    max_generation_length = 1024

    return ChatModel(
        generator,
        tokeniser,
        min_query_length,
        max_context_length,
        max_generation_length,
        chat_model_threads,
    )
