from pydantic_settings import BaseSettings


class Config(BaseSettings):
    server_port: int = 49494
    server_root_path: str = "/api"
    use_cuda: bool = False
    chat_model_threads: int = 1

    testing: bool = False

    postgres_endpoint: str = "postgres:5432"
