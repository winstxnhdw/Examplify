from fastapi.responses import StreamingResponse

from server.api.v1 import v1
from server.features import LLM
from server.schemas.v1 import Generate


@v1.post('/generate')
def generate(request: Generate) -> StreamingResponse:
    """
    Summary
    -------
    the `/generate` route provides an endpoint for generating text directly from the LLM model
    """
    prompts = (f'<s>[INST] {instruction} [\\INST]' for instruction in request.instructions)

    return StreamingResponse(
        LLM.generate(LLM.tokeniser(prompt).tokens() for prompt in prompts),
        media_type='text/event-stream'
    )
