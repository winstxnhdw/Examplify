from pydantic import BaseModel, Field


class Query(BaseModel):
    """
    Summary
    -------
    the query request

    Attributes
    ----------
    query (str) : the query
    """
    query: str = Field(examples=['Why did the chicken cross the road?'])
