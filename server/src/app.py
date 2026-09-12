from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from functools import partial
from logging import Logger, getLogger
from unittest.mock import create_autospec

from advanced_alchemy.extensions.litestar import (
    AsyncSessionConfig,
    SQLAlchemyAsyncConfig,
    SQLAlchemyPlugin,
    async_autocommit_handler_maker,
)
from litestar import Litestar, Response
from litestar.di import Provide
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Server
from litestar.serialization import decode_json, encode_json
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR
from litestar.types import ControllerRouterHandler
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.api import ChatController, health
from src.config import Config
from src.dependencies import limit_offset_pagination
from src.lifespans import load_chat_model


def exception_handler(logger: Logger, _, exception: Exception) -> Response[dict[str, str]]:
    logger.error(exception, exc_info=exception)

    return Response(
        content={"detail": "Internal Server Error"},
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
    )


def app() -> Litestar:
    config = Config()
    app_name = "Examplify"
    logger = getLogger(app_name)

    description = (
        "An offline CPU-first memory-scarce chat application to perform "
        "Retrieval-Augmented Generation (RAG) on your corpus of data"
    )

    openapi_config = OpenAPIConfig(
        title="Examplify",
        version="3.0.0",
        description=description,
        use_handler_docstrings=True,
        servers=[Server(url=config.server_root_path)],
    )

    create_engine_callable = partial(
        create_async_engine,
        json_serializer=encode_json,
        json_deserializer=decode_json,
        pool_use_lifo=True,
    )

    connection_string = None if config.testing else f"postgresql+asyncpg://postgres@{config.postgres_endpoint}/postgres"
    sqlalchemy_config = SQLAlchemyAsyncConfig(
        create_engine_callable=create_engine_callable,
        connection_string=connection_string,
        engine_instance=None if not config.testing else create_autospec(AsyncEngine),
        before_send_handler=async_autocommit_handler_maker(),
        session_config=AsyncSessionConfig(expire_on_commit=False),
        create_all=True,
    )

    route_handlers: list[ControllerRouterHandler] = [
        health,
        ChatController,
    ]

    lifespans: tuple[Callable[[Litestar], AbstractAsyncContextManager[None]], ...] = (
        load_chat_model(chat_model_threads=config.chat_model_threads, use_cuda=config.use_cuda),
    )

    return Litestar(
        openapi_config=openapi_config,
        exception_handlers={HTTP_500_INTERNAL_SERVER_ERROR: partial(exception_handler, logger)},
        route_handlers=route_handlers,
        dependencies={"limit_offset": Provide(limit_offset_pagination, sync_to_thread=False)},
        plugins=[SQLAlchemyPlugin(sqlalchemy_config)],
        lifespan=lifespans,
    )
