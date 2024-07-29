from redis import ResponseError
from redis.asyncio import Redis as RedisAsync
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from server.config import Config
from server.databases import Redis


async def try_create_index(client: RedisAsync, index_name: str):
    """
    Summary
    -------
    create a Redis index

    Parameters
    ----------
    client (RedisAsync) : a Redis client
    vector_dimensions (int) : number of vector dimensions
    """
    try:
        await client.ft(index_name).info()

    except ResponseError:
        vector_field = VectorField(
            'vector',
            'FLAT',
            {
                'TYPE': 'FLOAT32',
                'DISTANCE_METRIC': 'COSINE',
                'DIM': Config.embedding_dimensions,
            },
        )

        index_definition = IndexDefinition([Config.document_index_prefix], index_type=IndexType.HASH)

        await client.ft(index_name).create_index(
            fields=(TagField(Config.document_index_tag), vector_field),
            definition=index_definition,
        )


async def create_redis_index():
    """
    Summary
    -------
    initialise a Redis index
    """
    async with RedisAsync.from_pool(Redis.pool) as client:
        await try_create_index(client, Config.redis_index_name)
