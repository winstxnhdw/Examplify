from pydantic import BaseModel, Field


class DocumentSchema(BaseModel):
    """
    Summary
    -------
    the document schema

    Attributes
    ----------
    id (str) : the document id
    name (str) : the document name
    """
    id: str = Field(examples=['a8ae18389ed34ca391cdcc0160448df8'])
    name: str = Field(examples=["Stephen S. Carey - A Beginner's to Scientific Method, 4th Edition"])


class Uploaded(BaseModel):
    """
    Summary
    -------
    the upload file response schema

    Attributes
    ----------
    documents (list[DocumentSchema]) : the document schemas
    """
    documents: list[DocumentSchema | None]
