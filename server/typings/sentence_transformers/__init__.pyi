# pylint: skip-file

from typing import Iterable

from numpy import float64
from numpy.typing import NDArray
from torch.nn import Module, Sequential

class SentenceTransformer(Sequential):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules: Iterable[Module] | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
        use_auth_token: bool | str | None = None,
    ) -> None: ...
    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = False,
    ) -> NDArray[float64]: ...
