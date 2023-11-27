
from time import time

from server.api.debug import debug
from server.features import LLM
from server.features.llm.types import Message
from server.schemas.v1 import Benchmark, Query


@debug.post('/benchmark')
def benchmark(request: Query) -> Benchmark:
    """
    Summary
    -------
    the `/benchmark` route
    """
    message: Message = {
        'role': 'user',
        'content': request.query
    }

    prompt = LLM.tokeniser.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
    tokenised_prompt = LLM.tokeniser(prompt).tokens()

    start = time()
    response = ''.join(LLM.generate([tokenised_prompt]))
    total_time = time() - start

    output_tokens = LLM.tokeniser(response).tokens()
    total_tokens = len(tokenised_prompt) + len(LLM.static_prompt) + len(output_tokens)

    return Benchmark(
        response=response,
        tokens=total_tokens,
        total_time=total_time,
        tokens_per_second=total_tokens / total_time
    )
