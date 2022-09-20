from core.hyperparameters.bart_hyperparameters import BartHyperparametersV1

from transformers import BartTokenizer  # type: ignore


class BartFoCusTokenizerV1(BartTokenizer):
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
        hyperparameters: BartHyperparametersV1,
        **kwargs,
    ):

        tokenizer: BartTokenizer = BartTokenizer.from_pretrained(*args, **kwargs)

        if hyperparameters is not None:
            tokens = [
                hyperparameters.dialog_bos_token,
                hyperparameters.dialog_eos_token,
            ]

            tokenizer.add_special_tokens(
                {"additional_special_tokens": tokens},  # type: ignore
            )

        return tokenizer
