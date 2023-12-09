from typing import Awaitable, Callable

from server.features.llm import LLM
from server.features.llm.types import Message
from server.features.query.question_answering import question_answering


def query_llm(
    query: str,
    context: str,
    message_history: list[Message],
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

    Returns
    -------
    messages (list[Message]) : a list of messages
    """
    context_prompt = f'Given the following context:\n\n{context}\n\n' if context else ''

    message_history.append({
        'role': 'user',
        'content': f'{context_prompt}Please answer the following question:\n\n{query}'
    })

    return question_answering(message_history, LLM.query)
