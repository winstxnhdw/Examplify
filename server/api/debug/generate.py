from time import perf_counter

from litestar import Controller, post

from server.features.chat.types import Message
from server.schemas.v1 import Benchmark, Generate, Query
from server.state import AppState


class LLMController(Controller):
    """
    Summary
    -------
    Litestar controller for LLM-related debug endpoints
    """

    path = '/llm'

    @post()
    async def generate(self, state: AppState, data: Generate) -> str:
        """
        Summary
        -------
        an endpoint for generating text directly from the LLM model
        """
        chat = state.chat

        prompt = chat.tokeniser.apply_chat_template(
            [{'role': 'user', 'content': data.instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

        return await chat.generate(chat.tokeniser(prompt).tokens())

    @post('/benchmark')
    async def benchmark(self, state: AppState, data: Query) -> Benchmark:
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
        response = await chat.generate(tokenised_prompt)
        total_time = perf_counter() - start

        output_tokens = chat.tokeniser(response).tokens()
        total_tokens = len(tokenised_prompt) + len(chat) + len(output_tokens)

        return Benchmark(
            response=response,
            tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
        )
