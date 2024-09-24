from numpy import float32
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from torch import device

from examplify.features.embeddings.flag_embedding import FlagEmbedding
from examplify.utils import huggingface_download


class Embedder(SentenceTransformer):
    """
    Summary
    -------
    wrapper around a SentenceTransformer which routes the forward

    Methods
    -------
    encode_normalise(sentences: str | list[str]) -> bytes
        encode a sentence or list of sentences into a normalised embedding

    encode_query(sentence: str) -> bytes
        encode a sentence for searching relevant passages
    """

    def __init__(self):
        model_name = 'bge-base-en-v1.5'
        super().__init__(f'BAAI/{model_name}')
        self.cached_device = super().device  # type: ignore

        model_path = huggingface_download(f'winstxnhdw/{model_name}-ct2')
        self[0] = FlagEmbedding(self[0], model_path, 'auto')

    @property
    def device(self) -> device:
        return self.cached_device

    def encode_normalise(self, sentences: str | list[str], prompt: str | None = None) -> NDArray[float32]:
        """
        Summary
        -------
        encode a sentence or list of sentences into a normalised embedding

        Parameters
        ----------
        sentences (str | list[str]) : the sentence(s) to encode

        Returns
        -------
        embeddings (bytes) : the normalised embeddings
        """
        return self.encode(sentences, prompt=prompt, normalize_embeddings=True)

    def encode_query(self, sentence: str) -> NDArray[float32]:
        """
        Summary
        -------
        encode a sentence for searching relevant passages

        Parameters
        ----------
        sentence (str) : the sentence to encode

        Returns
        -------
        embeddings (bytes) : the normalised embeddings
        """
        return self.encode_normalise(sentence, 'Represent this sentence for searching relevant passages:')
