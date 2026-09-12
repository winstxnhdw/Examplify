from typing import Annotated
from uuid import UUID

from advanced_alchemy.filters import LimitOffset
from advanced_alchemy.service import OffsetPagination
from litestar import Controller, delete, get, post
from litestar.background_tasks import BackgroundTask
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.openapi.spec.example import Example
from litestar.params import Dependency, Parameter
from litestar.response import ServerSentEvent
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED

from src.dependencies import chat_store_service
from src.schemas import Chat, Query
from src.state import AppState
from src.stores.sql.services import ChatStoreService
from src.utils import exhaust, gather_any


class ChatController(Controller):
    path = "/chats"
    dependencies = {  # noqa: RUF012
        "chat_store": Provide(chat_store_service),
    }

    @get()
    async def get_chats(
        self,
        limit_offset: LimitOffset,
        chat_store: Annotated[ChatStoreService, Dependency()],
        *,
        only_indices: Annotated[bool, Parameter(query="only-indices", default=True)],
    ) -> OffsetPagination[Chat]:
        return await chat_store.get_chats(limit_offset, only_indices=only_indices)

    @get("/{chat_id:uuid}")
    async def get_chat(
        self,
        chat_store: Annotated[ChatStoreService, Dependency()],
        chat_id: Annotated[UUID, Parameter(examples=[Example(value="0195f17f-f01a-7347-be7e-9b5074881c2c")])],
    ) -> Chat:
        return await chat_store.get_chat(chat_id)

    @delete("/{chat_id:uuid}")
    async def delete_chat(
        self,
        chat_store: Annotated[ChatStoreService, Dependency()],
        chat_id: Annotated[UUID, Parameter(examples=[Example(value="0195f17f-f01a-7347-be7e-9b5074881c2c")])],
    ) -> None:
        if error := await chat_store.delete_chat(chat_id):
            raise NotFoundException from error

    @post("/{chat_id:uuid}/query")
    async def query(
        self,
        state: AppState,
        chat_id: Annotated[UUID, Parameter(examples=[Example(value="0195f17f-f01a-7347-be7e-9b5074881c2c")])],
        data: Query,
        event_type: Annotated[str | None, Parameter(query="event-type")] = None,
        *,
        store_query: Annotated[bool, Parameter(query="store-query")] = True,
    ) -> ServerSentEvent:
        session = state.session_maker_class()
        chat_store_generator = chat_store_service(db_session=session)
        chat_store = await anext(chat_store_generator)
        answer = chat_store.answer_query(
            state.chat_agent,
            chat_id,
            data.query,
            store_query=store_query,
        )

        answer = [a async for a in answer]

        return ServerSentEvent(
            answer,
            event_type=event_type,
            status_code=HTTP_201_CREATED if store_query else HTTP_200_OK,
            background=BackgroundTask(gather_any, session.__aexit__(None, None, None), exhaust(chat_store_generator)),
        )
