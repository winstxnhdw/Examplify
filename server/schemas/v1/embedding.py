from msgspec import Struct


class Embedding(Struct):
    """
    Summary
    -------
    the response schema on embedding

    Attributes
    ----------
    instruction (str) : the instruction
    text (str) : the text to compare similarity
    """

    instruction: str
    text: str
