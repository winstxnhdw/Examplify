from time import perf_counter

from litestar import Controller, post
from litestar.response import ServerSentEvent

from server.features.chat.types import Message
from server.schemas.v1 import Benchmark, Query
from server.state import AppState


class LLMController(Controller):
    """
    Summary
    -------
    Litestar controller for LLM-related debug endpoints
    """

    path = '/llm'

    @post(sync_to_thread=True)
    def generate(self, state: AppState, data: Query) -> ServerSentEvent:
        """
        Summary
        -------
        an endpoint for generating text directly from the LLM model
        """
        chat = state.chat

        prompt = chat.tokeniser.apply_chat_template(
            [{'role': 'user', 'content': data.query}],
            tokenize=False,
            add_generation_prompt=True,
        )

        return ServerSentEvent(chat.generate(chat.tokeniser(prompt).tokens()))

    @post('/benchmark', sync_to_thread=True)
    def benchmark(self, state: AppState, data: Query) -> Benchmark:
        """
        Summary
        -------
        an endpoint for benchmarking the LLM model
        """
        chat = state.chat
        message: Message = {'role': 'user', 'content': data.query}
        prompt = chat.tokeniser.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
        tokenised_prompt = chat.tokeniser(prompt).tokens()

        start = perf_counter()
        response = ''.join(chat.generate(tokenised_prompt))
        total_time = perf_counter() - start

        output_tokens = chat.tokeniser(response).tokens()
        total_tokens = len(tokenised_prompt) + len(chat) + len(output_tokens)

        return Benchmark(
            response='response',
            tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
        )
