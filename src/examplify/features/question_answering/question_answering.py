from typing import Callable, Iterator

from examplify.features.chat.types import Message


def question_answering(
    messages: list[Message],
    chain: Callable[[list[Message]], Iterator[str] | None],
) -> Iterator[str]:
    """
    Summary
    -------
    ask a question and get an answer

    Parameters
    ----------
    messages (list[Message]): the message history
    chain (Callable[[Sequence[Message]], Iterator[str] | None]): the callable to chain the messages

    Yields
    -------
    answer (str): token iterator of the answer
    """
    while (answer := chain(messages)) is None:
        messages = messages[1:]

    return answer
