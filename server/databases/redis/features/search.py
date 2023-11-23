from numpy import float64
from numpy.typing import NDArray
from redis.asyncio import Redis

from server.config import Config
from server.databases.redis.helpers import create_query_parameters
from server.databases.redis.helpers import redis_query as redis_query_helper


async def search(redis: Redis, search_field: str, embedding: NDArray[float64], top_k: int) -> str:
    """
    Summary
    -------
    retrieve the content of relevant documents

    Parameters
    ----------
    redis (Redis) : a Redis client
    search_field (str) : the field to search in
    embedding (NDArray[float64]) : the embedding to search for
    top_k (int) : the number of documents to retrieve

    Returns
    -------
    content (str) : the content of the relevant documents
    """
    redis_query = redis_query_helper('tag', search_field, top_k)
    redis_query_parameters = create_query_parameters(embedding)

    search_response = await redis.ft(Config.redis_index_name).search(
        redis_query,
        redis_query_parameters  # type: ignore  (this is a bug in the redis-py library)
    )

    return ' '.join(
        document['content'] for document
        in search_response.docs # type: ignore
    )
