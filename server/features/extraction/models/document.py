
from typing import NamedTuple


class Section(NamedTuple):
    """
    Summary
    -------
    a section of a document

    Attributes
    ----------
    link (str): the link to the section
    content (str): the content of the section
    """
    link: str
    content: str

class Document(NamedTuple):
    """
    Summary
    -------
    a document

    Attributes
    ----------
    id (str): the id of the document
    sections (list[Section]): the sections of the document
    semantic_identifier (str): the original name of the document
    """
    id: str
    sections: list[Section]
    semantic_identifier: str
