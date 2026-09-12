from collections.abc import AsyncIterator
from typing import Any


async def exhaust(iterator: AsyncIterator[Any]) -> None:
    async for _ in iterator:
        pass
