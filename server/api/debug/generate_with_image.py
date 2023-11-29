from fastapi import UploadFile

from server.api.debug import debug
from server.features import LLM, extract_text_from_image
from server.features.llm.types import Message
from server.schemas.v1 import Answer


@debug.post('/generate_with_image')
def generate_with_image(request: UploadFile) -> Answer:
    """
    Summary
    -------
    the `/generate_with_image` route provides an endpoint for generating text directly from the LLM model
    """
    messages: list[Message] = [{
        'role': 'user',
        'content': extract_text_from_image(request.file)
    }]

    prompt = LLM.tokeniser.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    messages.append({
        'role': 'assistant',
        'content': ''.join(LLM.generate([LLM.tokeniser(prompt).tokens()]))
    })

    return Answer(messages=messages)
