from json import dumps

from redis.asyncio import Redis

from server.features.llm.types import Message


async def save_messages(
    redis: Redis,
    chat_id: str,
    messages: list[Message],
    store_query: bool
) -> list[Message]:
    """
    Summary
    -------
    save a message in the database

    Parameters
    ----------
    redis (Redis) : a Redis client
    chat_id (str) : a chat id
    messages (list[Message]) : a list of messages
    store_query (bool) : whether or not to store the query in the database

    Returns
    -------
    messages (list[Message]) : a list of messages
    """
    if store_query:
        await redis.set(f'chat:{chat_id}', dumps(messages))

    return messages
