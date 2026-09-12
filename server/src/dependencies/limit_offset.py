from typing import Annotated

from advanced_alchemy.filters import LimitOffset
from litestar.params import Parameter


def limit_offset_pagination(
    current_page: Annotated[int, Parameter(ge=1, query="current-page", default=1, required=False)],
    page_size: Annotated[int, Parameter(query="page-size", ge=1, default=10, required=False)],
):
    return LimitOffset(page_size, page_size * (current_page - 1))
