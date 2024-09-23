from typing import Annotated

from msgspec import Meta, Struct


class Files(Struct):
    """
    Summary
    -------
    the response schema on file upload

    Attributes
    ----------
    indices (list[File]) : the document schemas
    """

    indices: Annotated[list[str], Meta(examples=['glsODUdnaUei_A_IxfSRI'])]
