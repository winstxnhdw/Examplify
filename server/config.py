from typing import Callable

from pydantic_settings import BaseSettings


def singleton[T](callable_object: Callable[[], T]) -> T:
    """
    Summary
    -------
    a decorator to transform a callable/class to a singleton

    Parameters
    ----------
    callable_object (Callable[[], T]) : the callable to transform

    Returns
    -------
    instance (T) : the singleton
    """
    return callable_object()


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
    chat_model_threads: int = 1

    document_index_prefix: str = 'doc'
    document_index_tag: str = 'chat_id'
    embedding_dimensions: int = 768
    minimum_similarity_score: float = 0.3

    redis_host: str = 'redis'
    redis_port: int = 6379
