from server.api.debug import debug
from server.features import LLM
from server.schemas.v1 import Generate


@debug.post('/generate')
async def generate(request: Generate) -> str:
    """
    Summary
    -------
    the `/generate` route provides an endpoint for generating text directly from the LLM model
    """
    prompt = LLM.tokeniser.apply_chat_template(
        [{'role': 'user', 'content': request.instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )

    return await LLM.generate(LLM.tokeniser(prompt).tokens())
