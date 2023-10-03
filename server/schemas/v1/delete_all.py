from pydantic import BaseModel, Field


class DeleteAll(BaseModel):
    """
    Summary
    -------
    the response schema for the `/delete_all` endpoint

    Attributes
    ----------
    timestamp (str) : the current timestamp
    """
    timestamp: str = Field(examples=['2021-10-10T10:10:10.000'])
