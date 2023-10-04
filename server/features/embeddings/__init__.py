from typing import Any

from huggingface_hub import snapshot_download
from numpy import float64
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from server.features.embeddings.flag_embedding import FlagEmbedding
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
        compute_type: ComputeTypes = 'float32',
        **kwargs: Any
    ):

        super().__init__('BAAI/bge-base-en-v1.5', *args, **kwargs)

        self[0] = FlagEmbedding(
            self[0],
            snapshot_download('winstxnhdw/bge-base-en-v1.5-ct2', local_files_only=True),
            compute_type=compute_type
        )


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


    def encode_query(self, sentence: str) -> NDArray[float64]:
        """
        Summary
        -------
        encode a sentence for searching relevant passages

        Parameters
        ----------
        sentence (str) : the sentence to encode

        Returns
        -------
        embeddings (NDArray[float64]) : the normalised embeddings
        """
        return self.encode(
            f'Represent this sentence for searching relevant passages: {sentence}',
            normalize_embeddings=False
        )
