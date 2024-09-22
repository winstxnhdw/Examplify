from litestar import Router

from server.api.v1.chat import ChatController
from server.api.v1.files_to_text import files_to_text
from server.api.v1.health import health

v1 = Router('/v1', tags=['v1'], route_handlers=[ChatController, health, files_to_text])
