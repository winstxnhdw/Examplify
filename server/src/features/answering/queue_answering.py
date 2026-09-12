from collections.abc import AsyncIterator

from src.features.chat import ChatAgentProtocol, Prompt, Prompts


def queue_answering(
    chat_agent: ChatAgentProtocol,
    prompts: Prompts[Prompt],
) -> AsyncIterator[str]:
    while not (stream := chat_agent.generate(prompts)):
        prompts.popleft()

    return stream
