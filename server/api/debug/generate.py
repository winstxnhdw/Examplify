from time import perf_counter

from litestar import Controller, post

from server.features.chat import Chat
from server.features.chat.types import Message
from server.schemas.v1 import Benchmark, Generate, Query


class LLMController(Controller):
    """
    Summary
    -------
    Litestar controller for LLM-related debug endpoints
    """

    path = '/llm'

    @post()
    async def generate(self, request: Generate) -> str:
        """
        Summary
        -------
        an endpoint for generating text directly from the LLM model
        """
        prompt = Chat.tokeniser.apply_chat_template(
            [{'role': 'user', 'content': request.instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

        return await Chat.generate(Chat.tokeniser(prompt).tokens())

    @post('/benchmark')
    async def benchmark(self, data: Query) -> Benchmark:
        """
        Summary
        -------
        an endpoint for benchmarking the LLM model
        """
        message: Message = {'role': 'user', 'content': data.query}

        prompt = Chat.tokeniser.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
        tokenised_prompt = Chat.tokeniser(prompt).tokens()

        start = perf_counter()
        response = await Chat.generate(tokenised_prompt)
        total_time = perf_counter() - start

        output_tokens = Chat.tokeniser(response).tokens()
        total_tokens = len(tokenised_prompt) + len(Chat.static_prompt) + len(output_tokens)

        return Benchmark(
            response=response,
            tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
        )
