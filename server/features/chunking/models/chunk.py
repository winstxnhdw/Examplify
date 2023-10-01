from typing import NamedTuple


class Chunk(NamedTuple):
    """
    Summary
    -------
    a chunk of text

    Attributes
    ----------
    content (str): the content of the chunk
    """
    content: str
