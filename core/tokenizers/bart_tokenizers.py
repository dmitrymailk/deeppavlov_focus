from typing import Optional

from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV3,
)

from transformers import BartTokenizer  # type: ignore


class BartFoCusTokenizerV2(BartTokenizer):
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
        hyperparameters: Optional[BartHyperparametersV3] = None,
        **kwargs,
    ):

        tokenizer: BartTokenizer = BartTokenizer.from_pretrained(*args, **kwargs)

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
