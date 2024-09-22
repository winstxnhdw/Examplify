from litestar import Router

from server.api.debug.generate import LLMController
from server.api.debug.redis import RedisController

debug = Router('/debug', tags=['debug'], route_handlers=[LLMController, RedisController])
