from json import dumps
from typing import Awaitable, Callable

from redis.asyncio import Redis

from server.features.llm.types import Message


async def throwaway(messages: list[Message]) -> list[Message]:
    """
    Summary
    -------
    a throwaway async function

    Parameters
    ----------
    messages (list[Message]) : a list of messages

    Returns
    -------
    messages (list[Message]) : a list of messages
    """
    return messages


def save_messages(
    redis: Redis,
    chat_id: str,
    store_query: bool
) -> Callable[[list[Message]], Awaitable[list[Message]]]:
    """
    Summary
    -------
    save a message in the database

    Parameters
    ----------
    redis (Redis) : a Redis client
    chat_id (str) : a chat id
    store_query (bool) : whether or not to store the query in the database

    Returns
    -------
    save (Callable[[list[Message]], Awaitable[list[Message]]]) : a function that save messages
    """
    if store_query:
        return throwaway

    async def save(messages: list[Message]) -> list[Message]:
        """
        Summary
        -------
        save a message in the database

        Parameters
        ----------
        messages (list[Message]) : a list of messages

        Returns
        -------
        messages (list[Message]) : a list of messages
        """
        redis.set(f'chat:{chat_id}', dumps(messages))
        return messages

    return save
