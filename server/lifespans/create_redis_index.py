from redis import Redis as RedisProtocol
from redis import ResponseError
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from server.config import Config
from server.databases import Redis


def try_create_index(client: RedisProtocol, vector_dimensions: int, index_name: str):
    """
    Summary
    -------
    create a Redis index

    Parameters
    ----------
    client (RedisProtocol) : a Redis client
    vector_dimensions (int) : number of vector dimensions
    """
    try:
        client.ft(index_name).info()
        print('Index already exists!')

    except ResponseError:
        vector_field = VectorField('vector', 'FLAT', {
            'TYPE': 'FLOAT32',
            'DISTANCE_METRIC': 'COSINE',
            'DIM': vector_dimensions,
        })

        client.ft(index_name).create_index(
            fields=(TagField('tag'), vector_field),
            definition=IndexDefinition(prefix=[Config.document_index_prefix], index_type=IndexType.HASH)
        )


def create_redis_index():
    """
    Summary
    -------
    initialise a Redis index
    """
    with RedisProtocol.from_pool(Redis.pool) as client:
        try_create_index(client, 768, Config.redis_index_name)
