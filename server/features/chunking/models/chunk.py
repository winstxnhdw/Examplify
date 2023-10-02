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
    id: int
    source_id: str
    content: str
