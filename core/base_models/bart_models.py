from typing import Optional

from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV1,
    BartHyperparametersV2,
)
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1

import torch
from torch import nn

from transformers import BartConfig, BartModel, BartPretrainedModel  # type: ignore
from transformers.modeling_outputs import Seq2SeqLMOutput


class BartLMV1(BartPretrainedModel):
    """
    Дефолтная модель для языкового моделирования
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
        labels: Optional[torch.Tensor],
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
            encoder_last_hidden_state=outputs[0],
        )


class BartLMV2(BartPretrainedModel):
    """
    Модель у которой следующий лосс
    loss = loss_LM + loss_persona_cls + loss_knowledge_cls
    где
        loss_LM - лосс языковой модели
        loss_persona_cls - лосс классификации персоны
        loss_knowledge_cls - лосс классификации knowledge candaites

    для классификации персоны берется SEP токен после персоны
    для классификации knowledge candidates берется SEP токен после klowledge candidates
    """

    def __init__(
        self,
        config: BartConfig,
        hyperparameters: BartHyperparametersV2,
        tokenizer: BartFoCusTokenizerV1,
    ) -> None:
        super().__init__(config=config)
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters

        self.model = BartModel(config=config)
        self.lm_head = nn.Linear(
            config.d_model,
            len(tokenizer),
            bias=False,
        )
        self.persona_head = nn.Linear(
            config.d_model,
            hyperparameters.persona_labels_amount,
            bias=False,
        )
        self.knowledge_head = nn.Linear(
            config.d_model,
            hyperparameters.knowledge_labels_amount,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids_labels: Optional[torch.Tensor],
        persona_grounding: Optional[torch.Tensor],
        knowledge_answer_index: Optional[torch.Tensor],
        persona_sep_index: Optional[torch.Tensor],
        knowledge_sep_index: Optional[torch.Tensor],
        dialog_bos_index: Optional[torch.Tensor],
        dialog_eos_index: Optional[torch.Tensor],
    ) -> Seq2SeqLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits: torch.Tensor = self.lm_head(outputs[0])

        loss = 0.0
        if input_ids_labels is not None:
            # copy from https://github.com/pkchat-focus/FoCus/blob/main/classification_modules.py#L462 # noqa: E501
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,  # type: ignore
            )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids_labels[..., 1:].contiguous()
            loss += loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if persona_grounding is not None:
            assert persona_sep_index is not None
            last_outputs = outputs[0]
            persona_sep_vectors = []
            for persona_sep_index_i, batch_item in zip(persona_sep_index, last_outputs):
                persona_sep_vector = batch_item[persona_sep_index_i]
                persona_sep_vectors.append(persona_sep_vector)

            persona_vector = torch.vstack(persona_sep_vectors)
            persona_logits = self.persona_head(persona_vector)
            loss_fct = nn.BCEWithLogitsLoss()
            persona_grounding = persona_grounding.type_as(persona_logits)
            loss += loss_fct(persona_logits, persona_grounding)

        if knowledge_answer_index is not None:
            assert knowledge_sep_index is not None
            last_outputs = outputs[0]
            knowledge_sep_vectors = []
            for knowledge_sep_index_i, batch_item in zip(
                knowledge_sep_index,
                last_outputs,
            ):
                knowledge_sep_vector = batch_item[knowledge_sep_index_i]
                knowledge_sep_vectors.append(knowledge_sep_vector)

            knowledge_vector = torch.vstack(knowledge_sep_vectors)

            knowledge_logits = self.knowledge_head(knowledge_vector)
            loss_fct = nn.CrossEntropyLoss()
            loss += loss_fct(knowledge_logits, knowledge_answer_index.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,  # type: ignore
            logits=logits,  # type: ignore
            encoder_last_hidden_state=outputs[0],
        )
