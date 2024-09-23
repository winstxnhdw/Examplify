from typing import Annotated

from litestar import Controller, delete, post
from litestar.di import Provide
from litestar.params import Dependency, Parameter

from server.config import Config
from server.databases.redis.wrapper import RedisAsync
from server.dependencies import embedder_model, redis_client
from server.features.embeddings import Embedder
from server.schemas.v1 import Query


class RedisController(Controller):
    """
    Summary
    -------
    Litestar controller for Redis-related debug endpoints
    """

    path = '/redis'
    dependencies = {
        'redis': Provide(redis_client),
        'embedder': Provide(embedder_model),
    }

    @delete()
    async def delete_index(self, redis: Annotated[RedisAsync, Dependency()], recreate: bool = False) -> None:
        """
        Summary
        -------
        an endpoint for deleting all files from the Redis database
        """
        await redis.delete_index(Config.redis_index_name)

        if recreate:
            await redis.create_index(
                Config.redis_index_name,
                Config.embedding_dimensions,
                Config.document_index_tag,
                [Config.document_index_prefix],
            )

    @post('/{chat_id:str}')
    async def search(
        self,
        redis: Annotated[RedisAsync, Dependency()],
        embedder: Annotated[Embedder, Dependency()],
        chat_id: str,
        data: Query,
        search_size: Annotated[int, Parameter(gt=0)] = 1,
    ) -> str:
        """
        Summary
        -------
        an endpoint for searching the Redis vector database
        """
        return await redis.search(chat_id, embedder.encode_query(data.query), search_size)
