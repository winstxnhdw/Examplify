from litestar.datastructures import State
from redis.asyncio import ConnectionPool

from server.features.chat.model import ChatModel


class AppState(State):
    redis_pool: ConnectionPool
    chat: ChatModel
