from pydantic import BaseModel, Field


class Generate(BaseModel):
    """
    Summary
    -------
    the `/generate` request model

    Attributes
    ----------
    instructions (list[str]) : instructions for the model
    """
    instructions: list[str] = Field(examples=[
        ['Why did the chicken cross the road?'],
    ])
