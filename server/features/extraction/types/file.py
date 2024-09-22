from typing import BinaryIO, TypedDict


class File(TypedDict):
    name: str
    data: BinaryIO
