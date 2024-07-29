from functools import lru_cache
from typing import Callable

from pydantic_settings import BaseSettings


def cache[T](user_function: Callable[[type[T]], T]) -> Callable[[type[T]], T]:
    """
    Summary
    -------
    a specific decorator to cache the result of a function

    Parameters
    ----------
    user_function (Callable[[type[T]], T]) : the function to cache

    Returns
    -------
    wrapper (Callable[[type[T]], T]) : the wrapper function
    """
    return lru_cache(maxsize=None)(user_function)


@cache
def singleton[T](cls: type[T]) -> T:
    """
    Summary
    -------
    a decorator to make a class a singleton

    Parameters
    ----------
    cls (type[T]) : the class to make a singleton

    Returns
    -------
    instance (T) : the singleton instance
    """
    return cls()


@singleton
class Config(BaseSettings):
    """
    Summary
    -------
    the general config class

    Attributes
    ----------
    port (int) : the port to run the server on
    server_root_path (str) : the root path for the server
    document_index_prefix (str) : the prefix for the document index
    redis_index_name (str) : the name of the Redis index
    use_cuda (bool) : whether to use CUDA for inference
    document_index_prefix (str) : the common document prefix for Redis Search
    document_index_tag (str) : the common index prefix for Redis Search
    embedding_dimensions (int) : the dimensions of the embedding vector space
    """

    server_port: int = 49494
    server_root_path: str = '/api'
    redis_index_name: str = 'index'
    use_cuda: bool = False
    document_index_prefix: str = 'doc'
    document_index_tag: str = 'chat_id'
    embedding_dimensions: int = 768
