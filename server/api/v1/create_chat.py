
from uuid import uuid4

from server.api.v1 import v1
from server.schemas.v1 import Chat


@v1.get('/create_chat')
async def create_chat() -> Chat:
    """
    Summary
    -------
    the `/create_chat` route provides an endpoint to retrieve a chat
    """
    return Chat(chat_id=''.join(str(uuid4()).split('-')))
