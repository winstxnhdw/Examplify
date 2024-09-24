from typing import Iterator, TypedDict

from ctranslate2 import Encoder, StorageView
from numpy import array
from torch import Tensor, as_tensor, device, float32, int32
from torch.nn import Module, Sequential

from examplify.types import ComputeTypes


class Features(TypedDict):
    """
    Summary
    -------
    a type hint for the features passed through the model
    """

    input_ids: Tensor
    attention_mask: Tensor
    token_embeddings: Tensor


class FlagEmbedding(Module):
    """
    Summary
    -------
    wrapper around a transformer model which routes the forward

    Attributes
    ----------
    transformer (Sequential) : the transformer model
    compute_type (ComputeTypes) : the compute type
    encoder (None) : the encoder model

    """

    def __init__(self, transformer: Sequential, model_path: str, compute_type: ComputeTypes):
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
        input_device: device = features['input_ids'].device

        if not self.encoder:
            self.encoder = Encoder(
                self.model_path,
                device=input_device.type,  # type: ignore
                device_index=input_device.index or 0,
                compute_type=self.compute_type,
            )

        input_indices = features['input_ids'].to(int32)
        length = features['attention_mask'].sum(1, dtype=int32)

        if input_device.type == 'cpu':
            input_indices = input_indices.numpy()
            length = length.numpy()

        outputs = self.encoder.forward_batch(StorageView.from_array(input_indices), StorageView.from_array(length))

        last_hidden_state = outputs.last_hidden_state

        if input_device.type == 'cpu':
            last_hidden_state = array(last_hidden_state)

        features['token_embeddings'] = as_tensor(last_hidden_state, device=input_device).to(float32)

        return features
