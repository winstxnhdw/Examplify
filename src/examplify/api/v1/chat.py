from io import BytesIO
from typing import Annotated

from litestar import Controller, delete, get, post, put
from litestar.concurrency import _run_sync_asyncio as run_sync
from litestar.datastructures import UploadFile
from litestar.di import Provide
from litestar.enums import RequestEncodingType
from litestar.exceptions import ClientException
from litestar.params import Body, Dependency, Parameter
from litestar.response import ServerSentEvent
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED

from examplify.databases.redis.features import store_chunks
from examplify.databases.redis.wrapper import RedisAsync
from examplify.dependencies import embedder_model, redis_client
from examplify.features.chunking import SentenceSplitter, chunk_document
from examplify.features.embeddings import Embedder
from examplify.features.extraction import extract_documents_from_pdfs
from examplify.features.question_answering import question_answering
from examplify.schemas.v1 import Chat, Files, Messages, Query
from examplify.state import AppState


class ChatController(Controller):
    """
    Summary
    -------
    Litestar controller for chat endpoints
    """

    path = '/chats'
    dependencies = {
        'redis': Provide(redis_client),
        'embedder': Provide(embedder_model),
    }

    @get()
    async def create_chat(self) -> Chat:
        """
        Summary
        -------
        an endpoint to get a unique chat id
        """
        return Chat()

    @get('/{chat_id:str}/messages')
    async def get_chat(self, redis: Annotated[RedisAsync, Dependency()], chat_id: str) -> Messages:
        """
        Summary
        -------
        an endpoint for getting all chat messages
        """
        return Messages(messages=await redis.get_messages(chat_id))

    @delete('/{chat_id:str}')
    async def delete_chat(self, redis: Annotated[RedisAsync, Dependency()], chat_id: str) -> None:
        """
        Summary
        -------
        an endpoint for deleting a chat
        """
        await redis.delete_chat(chat_id)

    @delete('/{chat_id:str}/messages')
    async def delete_chat_messages(self, redis: Annotated[RedisAsync, Dependency()], chat_id: str) -> None:
        """
        Summary
        -------
        an endpoint for deleting all chat messages
        """
        await redis.delete_chat_messages(chat_id)

    @delete('/{chat_id:str}/files/{file_id:str}')
    async def delete_chat_file(
        self,
        redis: Annotated[RedisAsync, Dependency()],
        chat_id: str,
        file_id: str,
    ) -> None:
        """
        Summary
        -------
        an endpoint for deleting a file belonging to a chat
        """
        await redis.delete_document(chat_id, file_id)

    @put('/{chat_id:str}/files')
    async def upload_files(
        self,
        state: AppState,
        redis: Annotated[RedisAsync, Dependency()],
        embedder: Annotated[Embedder, Dependency()],
        chat_id: str,
        data: Annotated[list[UploadFile], Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> Files:
        """
        Summary
        -------
        an endpoint for uploading files to a chat
        """
        text_splitter = SentenceSplitter(state.chat.tokeniser, chunk_size=128, chunk_overlap=0)
        responses = []

        chunk_generator = store_chunks(
            redis,
            chat_id,
            embedder.encode_normalise,
            extract_documents_from_pdfs([{'data': await file.read(), 'name': file.filename} for file in data]),
            chunk_document,
            text_splitter,
        )

        async for file_id, file_name in chunk_generator:
            if not file_name or not file_id:
                raise ClientException(detail='Invalid file name!', status_code=422)

            responses.append(file_id)

        return Files(indices=responses)

    @post('/{chat_id:str}/query')
    async def query(
        self,
        state: AppState,
        redis: Annotated[RedisAsync, Dependency()],
        embedder: Annotated[Embedder, Dependency()],
        chat_id: str,
        data: Query,
        search_size: Annotated[int, Parameter(ge=0)] = 0,
        store_query: bool = True,
    ) -> ServerSentEvent:
        """
        Summary
        -------
        the `/query` route provides an endpoint for performning retrieval-augmented generation
        """
        context = '' if not search_size else await redis.search(chat_id, embedder.encode_query(data.query), search_size)
        context_content = f'Given the following context:\n\n{context}\n\n' if context else ''
        message_history = await redis.get_messages(chat_id)
        message_history.append(
            {
                'role': 'user',
                'content': f'{context_content}{data.query}',
            }
        )

        answer = await run_sync(question_answering, message_history, state.chat.query)

        return ServerSentEvent(
            answer if not store_query else redis.save_messages(chat_id, answer, message_history),
            status_code=HTTP_201_CREATED if store_query else HTTP_200_OK,
        )
