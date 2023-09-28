from numpy import float64
from numpy.typing import NDArray

from server.features.embeddings.flag_embedding import FlagEmbedding


class Embeddings:

    model = FlagEmbedding(compute_type='auto')

    @classmethod
    def encode(cls, text_chunks: list[str]) -> NDArray[float64]:
        """
        Summary
        -------
        encode text into a vector

        Parameters
        ----------
        text (str) : the text to encode

        Returns
        -------
        vector (np.ndarray) : the vector representation of the text
        """
        return cls.model.encode(text_chunks, normalize_embeddings=True)

