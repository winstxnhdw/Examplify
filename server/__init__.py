from logging import getLogger

from litestar import Litestar, Response, Router
from litestar.datastructures import State
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Server
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR
from redis.asyncio import ConnectionPool

from server.api.debug import LLMController, RedisController
from server.api.v1 import ChatController, files_to_text, health
from server.config import Config
from server.lifespans import chat_model, create_redis_index, download_embeddings, download_nltk


def exception_handler(_, exception: Exception) -> Response[dict[str, str]]:
    """
    Summary
    -------
    the Litestar exception handler

    Parameters
    ----------
    request (Request) : the request
    exception (Exception) : the exception

    Returns
    -------
    response (Response[dict[str, str]]) : the response
    """
    getLogger('custom.access').error('', exc_info=exception)

    return Response(
        content={'detail': 'Internal Server Error'},
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
    )


def app() -> Litestar:
    """
    Summary
    -------
    the Litestar application
    """
    description = (
        'An offline CPU-first memory-scarce chat application to perform '
        'Retrieval-Augmented Generation (RAG) on your corpus of data'
    )

    openapi_config = OpenAPIConfig(
        title='Examplify',
        version='2.0.0',
        description=description,
        use_handler_docstrings=True,
        servers=[Server(url=Config.server_root_path)],
    )

    v1 = Router('/v1', tags=['v1'], route_handlers=[ChatController, health, files_to_text])
    debug = Router('/debug', tags=['debug'], route_handlers=[LLMController, RedisController])

    return Litestar(
        openapi_config=openapi_config,
        exception_handlers={HTTP_500_INTERNAL_SERVER_ERROR: exception_handler},
        route_handlers=[v1, debug],
        lifespan=[create_redis_index, download_embeddings, download_nltk, chat_model],
        state=State({'redis_pool': ConnectionPool(host=Config.redis_host, port=Config.redis_port)}),
    )
