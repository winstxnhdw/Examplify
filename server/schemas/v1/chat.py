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
    chat_id: str = Field(examples=['a8ae18389ed34ca391cdcc0160448df8'])
