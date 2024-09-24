from litestar.datastructures import State
from redis.asyncio import ConnectionPool

from examplify.features.chat.model import ChatModel


class AppState(State):
    """
    Summary
    -------
    the Litestar application state

    Attributes
    ----------
    redis_pool (ConnectionPool) : the global Redis connection pool
    chat (ChatModel) : the LLM chat model
    """

    redis_pool: ConnectionPool
    chat: ChatModel
