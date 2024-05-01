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
    query: str = Field(examples=['What is the definition of ADHD?'])
