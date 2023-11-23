from json import dumps

from redis.asyncio import Redis

from server.config import Config
from server.databases.redis import create_query_parameters, redis_get
from server.databases.redis import redis_query as redis_query_helper
from server.features import LLM, Embedding
from server.features.llm.types import Message
from server.features.query.question_answering import question_answering


async def query_llm(redis: Redis, query: str, chat_id: str, top_k: int, store_query: bool) -> list[Message]:
    """
    Summary
    -------
    the query feature provides a reusable query function for all types of queries

    Parameters
    ----------
    redis (Redis) : a Redis client
    query (str) : a query string
    chat_id (str) : a chat id
    top_k (int) : the number of documents to retrieve
    store_query (bool) : whether or not to store the query in the database

    Returns
    -------
    messages (list[Message]) : a list of messages
    """
    redis_query = redis_query_helper('tag', chat_id, top_k)

    redis_query_parameters = create_query_parameters(
        Embedding().encode_query(query)
    )

    search_response = await redis.ft(Config.redis_index_name).search(
        redis_query,
        redis_query_parameters  # type: ignore  (this is a bug in the redis-py library)
    )

    context = ' '.join(
        document['content'] for document
        in search_response.docs # type: ignore
    )

    message_history: list[Message] = await redis_get(redis, f'chat:{chat_id}', _ := [])
    message_history.append({
        'role': 'user',
        'content': f'Given the following context:\n\n{context}\n\nPlease answer the following question:\n\n{query}'
    })

    messages = question_answering(message_history, LLM.query)

    if not store_query:
        await redis.set(f'chat:{chat_id}', dumps(messages))

    return messages
