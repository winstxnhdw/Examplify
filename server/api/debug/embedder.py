from typing import Annotated

from litestar import Controller, post
from litestar.di import Provide
from litestar.params import Dependency
from litestar.status_codes import HTTP_200_OK

from server.dependencies import embedder_model
from server.features.embeddings import Embedder
from server.schemas.v1 import Embedding


class EmbeddingController(Controller):
    """
    Summary
    -------
    Litestar controller for embedding-related debug endpoints
    """

    path = '/embedding'
    dependencies = {'embedder': Provide(embedder_model)}

    @post(status_code=HTTP_200_OK, sync_to_thread=True)
    def generate(
        self,
        embedder: Annotated[Embedder, Dependency()],
        data: Embedding,
        encode_query: bool = False,
    ) -> float:
        """
        Summary
        -------
        an endpoint for generating text directly from the LLM model
        """
        instruction_embedding = (
            embedder.encode_query(data.instruction) if encode_query else embedder.encode_normalise(data.instruction)
        )

        return float(instruction_embedding @ embedder.encode_normalise(data.text).T)
