from fastapi.responses import StreamingResponse

from server.api.debug import debug
from server.features import LLM
from server.schemas.v1 import Generate


@debug.post('/generate')
def generate(request: Generate) -> StreamingResponse:
    """
    Summary
    -------
    the `/generate` route provides an endpoint for generating text directly from the LLM model
    """
    prompts = (
        LLM.tokeniser.apply_chat_template([{
            'role': 'user',
            'content': instruction
        }], tokenize=False, add_generation_prompt=True)
        for instruction in request.instructions
    )

    return StreamingResponse((f'{response}\n\n' for response in
        LLM.generate(LLM.tokeniser(prompt).tokens() for prompt in prompts)),
        media_type='text/event-stream'
    )
