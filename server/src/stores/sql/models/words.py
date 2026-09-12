from sqlalchemy import Index
from sqlalchemy.orm import Mapped, mapped_column

from src.stores.sql.models.base import Base
from src.utils import nameof


class WordModel(Base):
    __tablename__ = "words"

    id: Mapped[str] = mapped_column(primary_key=True)

    __table_args__ = (
        Index(
            "ix_word_trgm",
            id,
            postgresql_using="gin",
            postgresql_ops={nameof(f"{id=}"): "gin_trgm_ops"},
        ),
    )
