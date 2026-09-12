from typing import Annotated

from msgspec import Meta, Struct


class Query(Struct, kw_only=True):
    query: Annotated[str, Meta(examples=["What is the definition of ADHD?"])]
