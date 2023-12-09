from json import dumps, loads
from typing import NamedTuple, Sequence, TypedDict

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from server.config import Config
from server.databases.redis.helpers import redis_query_builder
from server.features.llm.types import Message
from server.lifespans.create_redis_index import try_create_index


class Mappings(TypedDict):
    """
    Summary
    -------
    a mapping of a document's vector to its content

    Attributes
    ----------
    vector (bytes) : the document's vector
    content (str) : the document's content
    chat_id (str) : the document's tag
    """
    vector: bytes
    content: str
    chat_id: str


class SearchResponse(NamedTuple):
    """
    Summary
    -------
    a response from a Redis search

    Attributes
    ----------
    docs (list[Mappings]) : the documents returned by the search
    """
    docs: list[Mappings]


class RedisAsyncWrapper:
    """
    Summary
    -------
    a wrapper for Redis

    Attributes
    ----------
    redis (Redis | Pipeline) : the Redis client

    Methods
    -------
    pipeline() -> Pipeline:
        create a Redis pipeline

    hset(source_id: str, chunk_id: int, mapping: Mappings) -> int:
        store a mapping in Redis

    search(search_field: str, embedding: NDArray[float64], top_k: int) -> str:
        retrieve the content of relevant documents

    save_messages(chat_id: str, messages: Sequence[str]):
        save a sequence of messages in Redis

    get_messages(chat_id: str) -> list[str]:
        retrieve a sequence of messages from Redis

    recreate_index():
        recreate the Redis index
    """
    def __init__(self, redis: Redis | Pipeline):

        self.redis = redis


    def pipeline(self) -> Pipeline:
        """
        Summary
        -------
        create a Redis pipeline

        Returns
        -------
        pipeline (Pipeline) : the Redis pipeline
        """
        return self.redis.pipeline()


    async def hset(self, source_id: str, chunk_id: int, mapping: Mappings) -> int:
        """
        Summary
        -------
        store a mapping in Redis

        Parameters
        ----------
        source_id (str) : a source identifier
        chunk_id (int) : a chunk identifier
        mapping (Mappings) : the mapping to store

        Returns
        -------
        status (int) : the status of the operation
        """
        return await self.redis.hset(
            f'{Config.document_index_prefix}:{source_id}:{chunk_id}',
            mapping=mapping  # type: ignore
        )


    async def search(self, search_field: str, embedding: bytes, top_k: int) -> str:
        """
        Summary
        -------
        retrieve the content of relevant documents

        Parameters
        ----------
        search_field (str) : the field to search in
        embedding (NDArray[float64]) : the embedding to search for
        top_k (int) : the number of documents to retrieve

        Returns
        -------
        content (str) : the content of the relevant documents
        """
        redis_query = redis_query_builder(Config.document_index_tag, search_field, top_k)

        redis_query_parameters = {
            'vec': embedding
        }

        search_response: SearchResponse = await self.redis.ft(Config.redis_index_name).search(
            redis_query,
            redis_query_parameters  # type: ignore  (this is a bug in the redis-py library)
        )

        return ' '.join(document['content'] for document in search_response.docs)


    async def save_messages(self, chat_id: str, messages: Sequence[Message]):
        """
        Summary
        -------
        save a sequence of messages in Redis

        Parameters
        ----------
        chat_id (str) : the chat ID
        messages (Sequence[Message]) : the messages to save
        """
        await self.redis.set(f'chat:{chat_id}', dumps(messages))


    async def get_messages(self, chat_id: str) -> list[Message]:
        """
        Summary
        -------
        retrieve a sequence of messages from Redis

        Parameters
        ----------
        chat_id (str) : the chat identifier

        Returns
        -------
        messages (list[Message]) : the messages
        """
        messages_json: str | None = await self.redis.get(f'chat:{chat_id}')
        return [] if not messages_json else loads(messages_json)


    async def get_chat_id(self, key: str) -> str | None:
        """
        Summary
        -------
        retrieve the chat identifier of a file

        Parameters
        ----------
        file_id (str) : the file identifier

        Returns
        -------
        chat_id (str | None) : the chat identifier
        """
        chat_id: bytes = await self.redis.hget(  # type: ignore
            key,
            Config.document_index_tag
        )

        return None if not chat_id else chat_id.decode('utf-8')


    async def delete_chat_messages(self, chat_id: str):
        """
        Summary
        -------
        delete all messages from a chat

        Parameters
        ----------
        chat_id (str) : the chat identifier
        """
        await self.redis.delete(f'chat:{chat_id}')


    async def delete_document(self, chat_id: str, file_id: str):
        """
        Summary
        -------
        delete a document of a specific chat

        Parameters
        ----------
        chat_id (str) : the chat identifier
        file_id (str) : the document identifier
        """
        async with self.pipeline() as pipeline:
            async for key in self.redis.scan_iter(f'{Config.document_index_prefix}:{file_id}:*'):
                document_chat_id = await self.get_chat_id(key)

                if not document_chat_id or document_chat_id != chat_id:
                    continue

                await pipeline.delete(key)

            await pipeline.execute()


    async def delete_chat(self, chat_id: str):
        """
        Summary
        -------
        delete a chat

        Parameters
        ----------
        chat_id (str) : the chat identifier
        """
        await self.delete_chat_messages(chat_id)

        async with self.pipeline() as pipeline:
            async for key in self.redis.scan_iter(f'{Config.document_index_prefix}:*'):
                document_chat_id = await self.get_chat_id(key)

                if not document_chat_id or document_chat_id != chat_id:
                    continue

                await pipeline.delete(key)

            await pipeline.execute()


    async def recreate_index(self):
        """
        Summary
        -------
        recreate the Redis index
        """
        await self.redis.ft(Config.redis_index_name).dropindex(True)
        await try_create_index(self.redis, Config.redis_index_name)
