from redis import ConnectionPool


class Redis:
    """
    a static class that contains the Redis connection pool
    """
    pool = ConnectionPool(host='redis', port=6379)
