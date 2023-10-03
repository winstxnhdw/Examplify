from typing import Any

from numpy import float64
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from server.features.embeddings.flag_embedding import FlagEmbedding
from server.helpers import huggingface_download
from server.types import ComputeTypes


class Embedding(SentenceTransformer):
    """
    Summary
    -------
    wrapper around a SentenceTransformer which routes the forward

    Methods
    -------
    encode_normalise(sentences: str | list[str]) -> NDArray[float64]
        encode a sentence or list of sentences into a normalised embedding
    """
    def __init__(
        self,
        *args: Any,
        compute_type: ComputeTypes = 'auto',
        **kwargs: dict[str, Any]
    ):

        super().__init__('BAAI/bge-large-en-v1.5', *args, **kwargs)
        model_path = huggingface_download('winstxnhdw/bge-large-en-v1.5-ct2-int8')
        self[0] = FlagEmbedding(self[0], model_path, compute_type=compute_type)


    def encode_normalise(self, sentences: str | list[str]) -> NDArray[float64]:
        """
        Summary
        -------
        encode a sentence or list of sentences into a normalised embedding

        Parameters
        ----------
        sentences (str | list[str]) : the sentence(s) to encode

        Returns
        -------
        embeddings (NDArray[float64]) : the normalised embeddings
        """
        return self.encode(sentences, normalize_embeddings=True)
