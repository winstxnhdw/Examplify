from typing import Any

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from server.features.embeddings.flag_embedding import FlagEmbedding
from server.types import ComputeTypes


class Embedding(SentenceTransformer):
    """
    Summary
    -------
    wrapper around a SentenceTransformer which routes the forward
    """
    def __init__(
        self,
        *args: Any,
        compute_type: ComputeTypes = 'default',
        **kwargs: dict[str, Any]
    ):
        super().__init__('BAAI/bge-base-en-v1.5', *args, **kwargs)
        model_path = snapshot_download('winstxnhdw/bge-base-en-v1.5-ct2')
        self[0] = FlagEmbedding(self[0], model_path, compute_type=compute_type)
