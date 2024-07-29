# pylint: skip-file

from typing import (
    Any,
    AsyncIterable,
    Callable,
    Generic,
    Iterable,
    Literal,
    Self,
    overload,
)

from numpy import float64
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import TypeVar

from server.types import ComputeTypes, Devices

LogProbability = TypeVar('LogProbability', default=None, bound=float | None)
Scores = TypeVar('Scores', bound=list[float])

class GenerationStepResult(Generic[LogProbability]):
    batch_id: int
    hypothesis_id: int
    is_last: bool
    log_prob: LogProbability
    step: int
    token: str
    token_id: int

class GenerationResult(Generic[Scores]):
    scores: Scores
    sequences: list[list[str]]
    sequences_ids: list[list[int]]

class AsyncGenerationResult(Generic[Scores]):
    def done(self) -> bool: ...
    def result(self) -> GenerationResult[Scores]: ...

class EncoderForwardOutput:
    last_hidden_state: Any
    pooler_output: Any

class Generator:
    device: Devices

    def __init__(
        self,
        model_path: str,
        device: Devices = 'cpu',
        *,
        device_index: int | list[int] = 0,
        compute_type: ComputeTypes = 'default',
        inter_threads: int = 1,
        intra_threads: int = 0,
        max_queued_batches: int = 0,
        flash_attention: bool = False,
        tensor_parallel: bool = False,
        files: object = None,
    ) -> None: ...
    @overload
    def generate_batch(
        self,
        start_tokens: list[list[str]],
        *,
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        asynchronous: Literal[False] = False,
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[False] = False,
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> list[GenerationResult[list[Any]]]: ...
    @overload
    def generate_batch(
        self,
        start_tokens: list[list[str]],
        *,
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        asynchronous: Literal[False] = False,
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[True],
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> list[GenerationResult[list[float]]]: ...
    @overload
    def generate_batch(
        self,
        start_tokens: list[list[str]],
        *,
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        asynchronous: Literal[True],
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[False] = False,
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> list[AsyncGenerationResult[list[Any]]]: ...
    @overload
    def generate_batch(
        self,
        start_tokens: list[list[str]],
        *,
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        asynchronous: Literal[True],
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[True],
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> list[AsyncGenerationResult[list[float]]]: ...
    @overload
    def generate_iterable(
        self,
        start_tokens: Iterable[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        asynchronous: Literal[False] = False,
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[False] = False,
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> Iterable[GenerationResult[list[Any]]]: ...
    @overload
    def generate_iterable(
        self,
        start_tokens: Iterable[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        asynchronous: Literal[False] = False,
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[True],
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> Iterable[AsyncGenerationResult[list[float]]]: ...
    @overload
    def generate_iterable(
        self,
        start_tokens: Iterable[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        asynchronous: Literal[True],
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[False] = False,
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> Iterable[AsyncGenerationResult[list[Any]]]: ...
    @overload
    def generate_iterable(
        self,
        start_tokens: Iterable[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        asynchronous: Literal[True],
        beam_size: int = 1,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        return_end_token: bool = False,
        max_length: int = 512,
        min_length: int = 0,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        include_prompt_in_result: bool = True,
        return_scores: Literal[True],
        return_alternatives: bool = False,
        min_alternative_expansion_prob: float = 0,
        sampling_topk: int = 0,
        sampling_topp: float = 0,
        sampling_temperature: float = 1,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> Iterable[GenerationResult[list[float]]]: ...
    @overload
    def generate_tokens(
        self,
        prompt: list[str] | list[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        max_length: int = 512,
        min_length: int = 0,
        sampling_topk: int = 1,
        sampling_topp: float = 1,
        sampling_temperature: float = 1,
        return_log_prob: Literal[False] = False,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> Iterable[GenerationStepResult]: ...
    @overload
    def generate_tokens(
        self,
        prompt: list[str] | list[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        max_length: int = 512,
        min_length: int = 0,
        sampling_topk: int = 1,
        sampling_topp: float = 1,
        sampling_temperature: float = 1,
        return_log_prob: Literal[True],
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        callback: Callable[[GenerationStepResult[float]], bool] | None = None,
    ) -> Iterable[GenerationStepResult[float]]: ...
    @overload
    def async_generate_tokens(
        self,
        prompt: list[str] | list[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        max_length: int = 512,
        min_length: int = 0,
        sampling_topk: int = 1,
        sampling_topp: float = 1,
        sampling_temperature: float = 1,
        return_log_prob: Literal[False] = False,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> AsyncIterable[GenerationStepResult]: ...
    @overload
    def async_generate_tokens(
        self,
        prompt: list[str] | list[list[str]],
        max_batch_size: int = 0,
        batch_type: str = 'examples',
        *,
        max_length: int = 512,
        min_length: int = 0,
        sampling_topk: int = 1,
        sampling_topp: float = 1,
        sampling_temperature: float = 1,
        return_log_prob: Literal[True],
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        disable_unk: bool = False,
        suppress_sequences: list[list[str]] | None = None,
        end_token: str | list[str] | list[int] | None = None,
        static_prompt: list[str] | None = None,
        cache_static_prompt: bool = True,
        callback: Callable[[GenerationStepResult], bool] | None = None,
    ) -> AsyncIterable[GenerationStepResult[float]]: ...

class StorageView:
    @classmethod
    def from_array(cls, array: NDArray[float64] | Tensor) -> Self: ...

class Encoder:
    def __init__(
        self,
        model_path: str,
        device: Devices = 'cpu',
        *,
        device_index: int | list[int] = 0,
        compute_type: ComputeTypes = 'default',
        inter_threads: int = 1,
        intra_threads: int = 0,
        max_queued_batches: int = 0,
        files: object = None,
    ) -> None: ...
    def forward_batch(
        self,
        inputs: list[list[str]] | list[list[int]] | StorageView,
        lengths: StorageView | None = None,
        token_type_ids: list[list[int]] | None = None,
    ) -> EncoderForwardOutput: ...
