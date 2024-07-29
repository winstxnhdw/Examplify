from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from torch import device

from server.features.embeddings.flag_embedding import FlagEmbedding


class Embedding(SentenceTransformer):
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

    def __init__(self, *, force_download: bool = False):
        model_name = 'bge-base-en-v1.5'
        super().__init__(f'BAAI/{model_name}')
        self.cached_device = super().device  # type: ignore

        model_path = snapshot_download(f'winstxnhdw/{model_name}-ct2', local_files_only=not force_download)
        self[0] = FlagEmbedding(self[0], model_path, 'auto')

    @property
    def device(self) -> device:
        return self.cached_device

    def encode_normalise(self, sentences: str | list[str]) -> bytes:
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
        return self.encode(sentences, normalize_embeddings=True).tobytes()

    def encode_query(self, sentence: str) -> bytes:
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
        return self.encode_normalise(f'Represent this sentence for searching relevant passages: {sentence}')
