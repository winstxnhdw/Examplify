from typing import Callable, Sequence

from server.features.llm.types import Message


def question_answering(
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
    while not (answer := chain(messages)):
        messages = messages[1:]

    return messages + [answer]
