from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Protocol, Self

from src.features.chat.prompts import Prompt


class ChatAgentProtocol(Protocol):
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    async def generate(self, prompts: Sequence[Prompt]) -> AsyncIterator[str] | None: ...
