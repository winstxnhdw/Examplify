from pydantic_settings import BaseSettings


class ConfigModel:
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


def singleton(cls: type[ConfigModel]) -> ConfigModel:
    """
    Summary
    -------
    a decorator to make a class a singleton

    Parameters
    ----------
    cls (type[ConfigModel]) : the class to make a singleton

    Returns
    -------
    instance (ConfigModel) : the singleton instance
    """
    return cls()


@singleton
class Config(ConfigModel, BaseSettings):
    """
    Summary
    -------
    the general config class
    """
