from typing import Annotated

from msgspec import Meta, Struct


class Query(Struct):
    """
    Summary
    -------
    the query request

    Attributes
    ----------
    query (str) : the query
    """

    query: Annotated[str, Meta(examples=['What is the definition of ADHD?'])]
