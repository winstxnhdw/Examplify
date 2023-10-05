from datetime import datetime

from pydantic import BaseModel, Field


class Timestamp(BaseModel):
    """
    Summary
    -------
    the response schema for a timestamp

    Attributes
    ----------
    timestamp (str) : the current timestamp
    """
    timestamp: str = Field(
        examples=['2021-10-10T10:10:10.000'],
        default_factory=lambda: datetime.now().isoformat()
    )
