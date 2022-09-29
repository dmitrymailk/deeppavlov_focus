from typing import Optional

from core.hyperparameters.t5_hyperparameters import T5HyperparametersV1

from transformers import T5Tokenizer  # type: ignore


class T5FoCusTokenizerV1(T5Tokenizer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        *args,
        hyperparameters: Optional[T5HyperparametersV1] = None,
        **kwargs,
    ):

        tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(*args, **kwargs)

        if hyperparameters is not None:
            tokens = [
                hyperparameters.response_bos_token,
                hyperparameters.response_eos_token,
                hyperparameters.query_bos_token,
                hyperparameters.query_eos_token,
                hyperparameters.sep_token,
            ]

            tokenizer.add_special_tokens(
                {"additional_special_tokens": tokens},  # type: ignore
            )

        return tokenizer
