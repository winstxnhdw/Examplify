from pydantic import BaseModel, Field


class Search(BaseModel):
    """
    Summary
    -------
    the query request

    Attributes
    ----------
    query (str) : the query
    index_name (str) : the index name
    """
    query: str = Field(examples=['Why did the chicken cross the road?'])
    top_k: int = Field(examples=[5], default=1)
