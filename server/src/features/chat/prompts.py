from collections import deque
from collections.abc import Iterable, Iterator
from typing import overload

from msgspec import Struct
from src.features.chat.role import Role


class Prompt(Struct, kw_only=True):
    role: Role
    content: str


class Prompts[T](deque[T]):
    __slots__ = ("system_prompt",)

    @overload
    def __init__(self, iterable: Iterable[T], maxlen: int | None = None) -> None: ...
    @overload
    def __init__(self, *, maxlen: int | None = None) -> None: ...
    def __init__(self, iterable: Iterable[T] | None = None, maxlen: int | None = None) -> None:
        super().__init__(iterable, maxlen)  # pyright: ignore [reportArgumentType]
        self.system_prompt = None

    def add_system_prompt(self, prompt: T) -> None:
        self.system_prompt = prompt

    def __iter__(self) -> Iterator[T]:
        if self.system_prompt is not None:
            yield self.system_prompt

        yield from super().__iter__()
