from io import BytesIO
from typing import Annotated

from litestar import Controller, delete, get, post, put
from litestar.datastructures import UploadFile
from litestar.di import Provide
from litestar.enums import RequestEncodingType
from litestar.exceptions import ClientException
from litestar.params import Body, Dependency, Parameter

from server.databases.redis.features import store_chunks
from server.databases.redis.wrapper import RedisAsync
from server.dependencies.redis import redis_client
from server.features.chunking import SentenceSplitter, chunk_document
from server.features.embeddings import Embedding
from server.features.extraction import extract_documents_from_pdfs
from server.features.question_answering import question_answering
from server.schemas.v1 import Answer, Chat, Files, Query
from server.state import AppState


class ChatController(Controller):
    """
    Summary
    -------
    Litestar controller for chat endpoints
    """

    path = '/chats'
    dependencies = {'redis': Provide(redis_client)}

    @get()
    async def create_chat(self) -> Chat:
        """
        Summary
        -------
        an endpoint to get a unique chat id
        """
        return Chat()

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
        chat_id: str,
        data: Annotated[list[UploadFile], Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> Files:
        """
        Summary
        -------
        an endpoint for uploading files to a chat
        """
        embedder = Embedding()
        text_splitter = SentenceSplitter(state.chat.tokeniser, chunk_size=128, chunk_overlap=0)
        responses = []

        chunk_generator = store_chunks(
            redis,
            chat_id,
            embedder,
            extract_documents_from_pdfs([{'data': BytesIO(await file.read()), 'name': file.filename} for file in data]),
            chunk_document,
            text_splitter,
        )

        async for file_id, file_name in chunk_generator:
            if not file_name or not file_id:
                raise ClientException(detail='Invalid file name!', status_code=422)

            responses.append(file_id)

        return Files(documents=responses)

    @post('/{chat_id:str}/query')
    async def query(
        self,
        state: AppState,
        redis: Annotated[RedisAsync, Dependency()],
        chat_id: str,
        data: Query,
        search_size: Annotated[int, Parameter(ge=0)] = 0,
        store_query: bool = True,
    ) -> Answer:
        """
        Summary
        -------
        the `/query` route provides an endpoint for performning retrieval-augmented generation
        """
        context = (
            '' if not search_size else await redis.search(chat_id, Embedding().encode_query(data.query), search_size)
        )

        message_history = await redis.get_messages(chat_id)
        messages = await question_answering(data.query, context, message_history, state.chat.query)

        if store_query:
            await redis.save_messages(chat_id, messages)

        return Answer(messages=messages)
