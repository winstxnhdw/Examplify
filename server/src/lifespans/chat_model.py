from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from litestar import Litestar

from src.features.chat import get_ctranslate_model


@asynccontextmanager
async def chat_model_lifespan(
    app: Litestar,
    *,
    chat_model_threads: int,
    use_cuda: bool,
) -> AsyncIterator[None]:
    async with get_ctranslate_model(chat_model_threads, use_cuda=use_cuda) as chat_model:
        app.state.chat_agent = chat_model
        yield
        del app.state.chat_agent


def load_chat_model(
    *,
    chat_model_threads: int,
    use_cuda: bool,
) -> Callable[[Litestar], AbstractAsyncContextManager[None]]:
    return lambda app: chat_model_lifespan(
        app,
        chat_model_threads=chat_model_threads,
        use_cuda=use_cuda,
    )
