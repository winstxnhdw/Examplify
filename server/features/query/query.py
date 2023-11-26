from typing import Awaitable, Callable, Iterable

from server.features.llm import LLM
from server.features.llm.types import Message


def question_answering(
    messages: list[Message],
    chain: Callable[[Iterable[Message]], Message | None]
) -> list[Message]:
    """
    Summary
    -------
    ask a question and get an answer

    Parameters
    ----------
    messages (list[Message]): the message history
    chain (Callable[[Iterable[Message]], Message | None]): the model

    Returns
    -------
    messages (list[Message]): the message history
    """
    while not (answer := chain(messages)):
        messages = messages[1:]

    return messages + [answer]


async def query_llm(
    query: str,
    context: str,
    message_history: list[Message],
    save_messages: Callable[[list[Message]], Awaitable[list[Message]]]
) -> list[Message]:
    """
    Summary
    -------
    the query feature provides a reusable query function for all types of queries

    Parameters
    ----------
    query (str) : a query
    context (str) : the context to the query
    message_history (list[Message]) : a list of messages
    save_messages (Callable[[list[Message]], Awaitable[list[Message]]]) : a function that save messages

    Returns
    -------
    messages (list[Message]) : a list of messages
    """
    context_prompt = f'Given the following context:\n\n{context}\n\n' if context else ''

    message_history.append({
        'role': 'user',
        'content': f'{context_prompt}Please answer the following question:\n\n{query}'
    })

    return await save_messages(question_answering(message_history, LLM.query))
