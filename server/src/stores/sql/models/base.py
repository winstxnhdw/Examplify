from advanced_alchemy.base import AdvancedDeclarativeBase, CommonTableAttributes
from advanced_alchemy.mixins import SentinelMixin
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import MappedAsDataclass


class Base(MappedAsDataclass, AsyncAttrs, SentinelMixin, CommonTableAttributes, AdvancedDeclarativeBase, kw_only=True):
    __abstract__ = True
