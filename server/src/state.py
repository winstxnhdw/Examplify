from collections.abc import Callable

from litestar.datastructures import State
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.features.chat import ChatAgentProtocol


class AppState(State):
    chat_agent: ChatAgentProtocol
    session_maker_class: Callable[[], AsyncSession]
    db_engine: AsyncEngine
