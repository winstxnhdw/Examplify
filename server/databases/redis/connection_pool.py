from redis.asyncio import ConnectionPool


class Redis:
    """
    a static class that contains the async Redis connection pool

    Attributes
    ----------
    pool (ConnectionPool) : the async Redis connection pool
    """
    pool = ConnectionPool(host='redis', port=6379)
