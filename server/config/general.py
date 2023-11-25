from typing import TypeVar

from pydantic_settings import BaseSettings

T = TypeVar('T', bound=BaseSettings)


def singleton(cls: type[T]) -> T:
    """
    Summary
    -------
    a decorator to make a class a singleton

    Parameters
    ----------
    cls (type[T]) : the class to make a singleton

    Returns
    -------
    cls (T) : the singleton class
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
    worker_count (int) : the number of workers to use
    document_index_prefix (str) : the prefix for the document index
    redis_index_name (str) : the name of the Redis index
    """
    server_port: int = 49494
    server_root_path: str = '/api'
    worker_count: int = 1
    document_index_prefix: str = 'doc:'
    redis_index_name: str = 'index'
