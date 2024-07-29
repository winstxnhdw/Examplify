from pydantic import BaseModel, Field


class Chat(BaseModel):
    """
    Summary
    -------
    the chat response schema

    Attributes
    ----------
    chat_id (str) : the chat id
    """

    chat_id: str = Field(examples=['6a67cb2e-a618-46b1-b617-73a0f3805122'])
