from asyncio import gather
from collections.abc import Coroutine
from typing import Any


async def gather_any(*coroutines: Coroutine[Any, Any, Any]) -> Any:  # noqa: ANN401
    return await gather(*coroutines)
