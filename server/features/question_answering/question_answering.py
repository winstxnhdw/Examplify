from typing import Callable, Sequence

from server.features.llm.types import Message


def question_answering(
    query: str,
    context: str,
    messages: list[Message],
    chain: Callable[[Sequence[Message]], Message | None]
) -> list[Message]:
    """
    Summary
    -------
    ask a question and get an answer

    Parameters
    ----------
    messages (list[Message]): the message history
    chain (Callable[[Sequence[Message]], Message | None]): the model

    Returns
    -------
    messages (list[Message]): the message history
    """
    context_prompt = f'Given the following context:\n\n{context}\n\n' if context else ''

    messages.append({
        'role': 'user',
        'content': f'{context_prompt}Please answer the following question:\n\n{query}'
    })

    while not (answer := chain(messages)):
        messages = messages[1:]

    return messages + [answer]
