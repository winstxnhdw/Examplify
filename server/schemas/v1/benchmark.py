from pydantic import BaseModel


class Benchmark(BaseModel):
    """
    Summary
    -------
    the response schema for a benchmark request

    Attributes
    ----------
    response (str) : the response
    tokens_per_second (float) : the number of tokens generated per second
    """
    response: str
    tokens: int
    total_time: float
    tokens_per_second: float
