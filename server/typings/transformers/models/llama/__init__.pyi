# pylint: skip-file

from os import PathLike
from typing import Any, Iterable, Literal, Mapping, Self, overload

from transformers.pipelines.conversational import Conversation
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTokenizedInput, TextInput,
                                                  TruncationStrategy)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import PaddingStrategy, TensorType

class LlamaTokenizerFast(PreTrainedTokenizerFast):
    def __call__(
        self,
        text: str | PreTokenizedInput | list[PreTokenizedInput] | None = None,
        text_pair: TextInput | PreTokenizedInput | list[PreTokenizedInput] | None = None,
        text_target: TextInput | PreTokenizedInput | list[PreTokenizedInput] | None = None,
        text_pair_target: TextInput | PreTokenizedInput | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs: Any
    ) -> BatchEncoding: ...


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *init_inputs: Any,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any
    ) -> Self: ...


    @overload
    def apply_chat_template(
        self,
        conversation: Iterable[Mapping[str, Any]] | Conversation,
        chat_template: str | None = None,
        tokenize: Literal[False] = False,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        **tokenizer_kwargs: Any
    ) -> str: ...


    @overload
    def apply_chat_template(
        self,
        conversation: Iterable[Mapping[str, Any]] | Conversation,
        chat_template: str | None = None,
        tokenize: Literal[True] = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        **tokenizer_kwargs: Any
    ) -> list[int]: ...


    def apply_chat_template(
        self,
        conversation: Iterable[Mapping[str, Any]] | Conversation,
        chat_template: str | None = None,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        **tokenizer_kwargs: Any
    ) -> str | list[int]: ...


    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None
    ) -> tuple[str]: ...