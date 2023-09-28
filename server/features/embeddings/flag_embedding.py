from array import array
from typing import Any, Iterator, TypedDict

from ctranslate2 import Encoder, StorageView
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from torch import as_tensor, float32, get_num_threads, int32
from torch.nn import Module, Sequential

from server.types import ComputeTypes


class Features(TypedDict):
    """
    Summary
    -------
    a type hint for the features passed through the model
    """
    input_ids: Any
    attention_mask: Any
    token_type_ids: Any
    token_embeddings: Any


class CT2Transformer(Module):
    """
    Summary
    -------
    wrapper around a sentence_transformers.models.Transformer which routes the forward

    Attributes
    ----------
    transformer (Sequential) : the transformer model
    compute_type (ComputeTypes) : the compute type
    encoder (None) : the encoder model

    """
    def __init__(
        self,
        transformer: Sequential,
        model_path: str,
        compute_type: ComputeTypes = 'default'
    ):
        super().__init__()

        self.compute_type: ComputeTypes = compute_type
        self.encoder: Encoder | None = None
        self.tokenize = transformer.tokenize
        self.model_path = model_path


    def children(self) -> Iterator[Module]:
        return iter([])


    def forward(self, features: Features) -> Features:
        """
        Summary
        -------
        forward pass

        Parameters
        ----------
        features (Features) : the features to pass through the model

        Returns
        -------
        features (Features) : the features after passing through the model
        """
        device = features["input_ids"].device

        if not self.encoder:
            self.encoder = Encoder(
                self.model_path,
                device=device.type,
                device_index=device.index or 0,
                intra_threads=get_num_threads(),
                compute_type=self.compute_type,
            )

        input_indices = features['input_ids'].to(int32)

        length = features['attention_mask'].sum(1, dtype=int32)

        if device.type == 'cpu':
            input_indices = input_indices.numpy()
            length = length.numpy()

        input_indices = StorageView.from_array(input_indices)

        length = StorageView.from_array(length)

        outputs = self.encoder.forward_batch(input_indices, length)

        last_hidden_state = outputs.last_hidden_state

        if device.type == 'cpu':
            last_hidden_state = array(last_hidden_state)

        features['token_embeddings'] = as_tensor(last_hidden_state, device=device).to(float32)

        return features


class FlagEmbedding(SentenceTransformer):
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
        self[0] = CT2Transformer(self[0], model_path, compute_type=compute_type)
