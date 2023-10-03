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
    id: str = Field(examples=['6a67cb2e-a618-46b1-b617-73a0f3805122'])
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
