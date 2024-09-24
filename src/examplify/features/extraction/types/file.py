from typing import TypedDict


class File(TypedDict):
    """
    Summary
    -------
    the file type

    Attributes
    ----------
    name (str): the name of the file
    data (BinaryIO): the file data
    """

    name: str
    data: bytes
