from pydantic import BaseModel, Field


class Generate(BaseModel):
    """
    Summary
    -------
    the `/generate` request model

    Attributes
    ----------
    instruction (str) : instructions for the model
    """

    instruction: str = Field(examples=['Why did the chicken cross the road?'])
