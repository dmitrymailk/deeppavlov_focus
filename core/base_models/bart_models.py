from core.hyperparameters.bart_hyperparameters import BartHyperparametersV1
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1

import torch
from torch import nn

from transformers import BartConfig, BartModel, BartPretrainedModel  # type: ignore
from transformers.modeling_outputs import Seq2SeqLMOutput


class BartLMV1(BartPretrainedModel):
    # fmt: off
    r"""
    Simple usage:
        model = BartLMV1(
            config=BartConfig.from_pretrained('facebook/bart-large'),
            hyperparameters=BartFoCusDatasetSampleHyperparametersV1(),
            tokenizer=BartFoCusTokenizerV1.from_pretrained(
                'facebook/bart-base',
                hyperparameters=BartFoCusDatasetSampleHyperparametersV1()),
        )

        input_ids = torch.tensor([[1, 2, ]])
        attention_mask = torch.tensor([[1, 1,]])
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
    """
    # fmt: on

    def __init__(
        self,
        config: BartConfig,
        hyperparameters: BartHyperparametersV1,
        tokenizer: BartFoCusTokenizerV1,
    ) -> None:
        super().__init__(config=config)
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters

        self.model = BartModel(config=config)
        self.lm_head = nn.Linear(config.d_model, len(tokenizer), bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits: torch.Tensor = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # copy from https://github.com/pkchat-focus/FoCus/blob/main/classification_modules.py#L462 # noqa: E501
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,  # type: ignore
            )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,  # type: ignore
            # не очень понимаю что это за ключ в контексте модели BART
            encoder_last_hidden_state=outputs[0],
        )
