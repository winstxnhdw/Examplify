from huggingface_hub import snapshot_download
from numpy import float64
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from server.features.embeddings.flag_embedding import FlagEmbedding


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
    def __init__(self, *, force_download: bool = False):

        model_name = 'bge-base-en-v1.5'
        super().__init__(f'BAAI/{model_name}')

        model_path = snapshot_download(f'winstxnhdw/{model_name}-ct2', local_files_only=not force_download)
        self[0] = FlagEmbedding(self[0], model_path, 'auto')


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
        return self.encode_normalise(
            f'Represent this sentence for searching relevant passages: {sentence}'
        )
