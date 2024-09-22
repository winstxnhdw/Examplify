from litestar.datastructures import State
from redis.asyncio import ConnectionPool


class AppState(State):
    redis_pool: ConnectionPool
