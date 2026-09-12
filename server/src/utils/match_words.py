from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from re import Pattern
from re import compile as re_compile


def match_words[T](wrapper: Callable[[Pattern[str]], T]) -> T:
    return wrapper(re_compile(r"[\w'-]+"))


async def async_substitution(
    pattern: Pattern[str],
    text: str,
    transformer: Callable[[str], Awaitable[str]],
) -> str:
    new_words: deque[str] = deque()
    last_index = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        word = match.group()

        new_words.extend((text[last_index:start], await transformer(word)))
        last_index = end

    new_words.append(text[last_index:])

    return "".join(new_words)


@match_words
def extract_words(pattern: Pattern[str]) -> Callable[[str], Iterator[str]]:
    return lambda text: (match.group() for match in pattern.finditer(text))


@match_words
def replace_words_async(pattern: Pattern[str]) -> Callable[[str, Callable[[str], Awaitable[str]]], Awaitable[str]]:
    return lambda text, transformer: async_substitution(pattern, text, transformer)
