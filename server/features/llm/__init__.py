from typing import Generator, Iterable

from ctranslate2 import Generator as LLMGenerator
from huggingface_hub import snapshot_download
from transformers.models.llama import LlamaTokenizerFast


class LLM:
    """
    Summary
    -------
    a static class for generating text with an Large Language Model

    Methods
    -------
    generate(prompt: str) -> Generator[str, None, None]
        generate text from a prompt
    """
    model_path = snapshot_download('winstxnhdw/Luban-13B-ct2-int8')
    generator = LLMGenerator(model_path, device='cpu', compute_type='auto', inter_threads=1)
    tokeniser: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(model_path)
    suppress_sequences = [['<s>', '▁###', '▁Inst', 'ruction', ':']]
    stop_generator = False
    max_generation_length = 512

    @classmethod
    def stop_generation(cls):
        """
        Summary
        -------
        stop the generation of text
        """
        cls.stop_generator = True


    @classmethod
    def generate(cls, prompts: list[str] | Iterable[str]) -> Generator[str, None, None]:
        """
        Summary
        -------
        generate text from a series/single prompt(s)

        Parameters
        ----------
        prompt (str) : the prompt to generate text from

        Yields
        -------
        answer (str) : the generated answer
        """
        cls.stop_generator = False

        yield from (
            cls.tokeniser.decode(result.sequences_ids[0]) for result in cls.generator.generate_iterable(
                (cls.tokeniser(prompt).tokens() for prompt in prompts),
                length_penalty=1.1,
                repetition_penalty=1.2,
                suppress_sequences=cls.suppress_sequences,
                max_length=cls.max_generation_length,
                include_prompt_in_result=False,
                sampling_topk=20,
                sampling_topp=0.95,
                sampling_temperature=0.6,
                callback=lambda _: cls.stop_generator
            )
        )
