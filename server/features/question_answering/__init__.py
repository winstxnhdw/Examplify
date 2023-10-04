from collections import deque
from typing import Callable, Iterable

from server.features.llm.types import Message


def question_answering(
    question: str,
    context: str,
    message_history: deque[Message],
    chain: Callable[[Iterable[Message]], Message | None]
) -> deque[Message]:
    """
    Summary
    -------
    ask a question and get an answer

    Parameters
    ----------
    question (str): the question to ask
    context (str): the context to ask the question in
    message_history (deque[Message]): the message history

    Returns
    -------
    deque[Message]: the message history
    """
    message_history.append({
        'role': 'user',
        'content': f'Given the following context:\n\n{context}\n\nPlease answer the following question:\n\n{question}'
    })

    while not (answer := chain(message_history)):
        message_history.popleft()
        message_history.popleft()

    message_history.append(answer)

    return message_history
