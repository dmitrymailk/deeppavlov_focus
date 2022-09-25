from typing import Optional

from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV1,
    BartHyperparametersV2,
    BartHyperparametersV3,
)
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1, BartFoCusTokenizerV2

import torch
from torch import nn

from transformers import BartConfig, BartModel, BartPretrainedModel  # type: ignore
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import ModelOutput  # type: ignore


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


class BartLMV2Outputs(ModelOutput):
    loss: Optional[torch.Tensor]
    lm_loss: Optional[torch.Tensor]
    persona_loss: Optional[torch.Tensor]
    knowledge_loss: Optional[torch.Tensor]

    lm_logits: torch.Tensor
    persona_logits: torch.Tensor
    knowledge_logits: torch.Tensor
    last_hidden_state: torch.Tensor


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
        persona_sep_index: torch.Tensor,
        knowledge_sep_index: torch.Tensor,
        dialog_bos_index: Optional[torch.Tensor],
        dialog_eos_index: Optional[torch.Tensor],
    ) -> BartLMV2Outputs:
        assert persona_sep_index is not None
        assert knowledge_sep_index is not None

        bart_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_outputs: torch.Tensor = bart_outputs[0]
        lm_logits: torch.Tensor = self.lm_head(last_outputs)
        loss: torch.Tensor = torch.tensor(
            0,
            device=self.device,
            dtype=torch.float32,
        )
        persona_loss = None
        knowledge_loss = None
        lm_loss = None

        # compute lm loss
        if input_ids_labels is not None:
            # copy from https://github.com/pkchat-focus/FoCus/blob/main/classification_modules.py#L462 # noqa: E501
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,  # type: ignore
            )
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids_labels[..., 1:].contiguous()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss += lm_loss

        # extract persona vectors
        persona_sep_vectors = []
        for persona_sep_index_i, batch_item in zip(persona_sep_index, last_outputs):
            persona_sep_vector = batch_item[persona_sep_index_i]
            persona_sep_vectors.append(persona_sep_vector)

        persona_vector = torch.vstack(persona_sep_vectors)
        persona_logits = self.persona_head(persona_vector)

        # compute persona loss
        if persona_grounding is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            persona_grounding = persona_grounding.type_as(persona_logits)
            persona_loss = loss_fct(persona_logits, persona_grounding)
            loss += persona_loss

        # extract knowledge vectors
        knowledge_sep_vectors = []
        for knowledge_sep_index_i, batch_item in zip(
            knowledge_sep_index,
            last_outputs,
        ):
            knowledge_sep_vector = batch_item[knowledge_sep_index_i]
            knowledge_sep_vectors.append(knowledge_sep_vector)

        knowledge_vector = torch.vstack(knowledge_sep_vectors)
        knowledge_logits = self.knowledge_head(knowledge_vector)

        # compute knowledge loss
        if knowledge_answer_index is not None:
            loss_fct = nn.CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits, knowledge_answer_index.view(-1))
            loss += knowledge_loss

        return BartLMV2Outputs(
            # loss
            loss=loss,
            lm_loss=lm_loss,
            persona_loss=persona_loss,
            knowledge_loss=knowledge_loss,
            # logits
            lm_logits=lm_logits,
            persona_logits=persona_logits,
            knowledge_logits=knowledge_logits,
            last_hidden_state=bart_outputs[0],
        )


class BartLMV3(BartPretrainedModel):
    def __init__(
        self,
        config: BartConfig,
        hyperparameters: BartHyperparametersV2,
        tokenizer: BartFoCusTokenizerV1,
    ) -> None:
        """
        Модель у которой следующий лосс
        loss = loss_LM
        где
            loss_LM - лосс языковой модели

        для классификации персоны берется SEP токен после персоны

        для классификации knowledge candidates берется SEP токен после klowledge
        candidates

        эта модель нужна чтобы проверить что лосс persona и knowledge не влияют на лосс
        языковой модели
        """
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
        persona_sep_index: torch.Tensor,
        knowledge_sep_index: torch.Tensor,
        dialog_bos_index: Optional[torch.Tensor],
        dialog_eos_index: Optional[torch.Tensor],
    ) -> BartLMV2Outputs:
        assert persona_sep_index is not None
        assert knowledge_sep_index is not None

        bart_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_outputs: torch.Tensor = bart_outputs[0]
        lm_logits: torch.Tensor = self.lm_head(last_outputs)
        loss: torch.Tensor = torch.tensor(
            0,
            device=self.device,
            dtype=torch.float32,
        )
        persona_loss = None
        knowledge_loss = None
        lm_loss = None

        # compute lm loss
        if input_ids_labels is not None:
            # copy from https://github.com/pkchat-focus/FoCus/blob/main/classification_modules.py#L462 # noqa: E501
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,  # type: ignore
            )
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids_labels[..., 1:].contiguous()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss += lm_loss

        # extract persona vectors
        persona_sep_vectors = []
        for persona_sep_index_i, batch_item in zip(persona_sep_index, last_outputs):
            persona_sep_vector = batch_item[persona_sep_index_i]
            persona_sep_vectors.append(persona_sep_vector)

        persona_vector = torch.vstack(persona_sep_vectors)
        persona_logits = self.persona_head(persona_vector)

        # compute persona loss
        if persona_grounding is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            persona_grounding = persona_grounding.type_as(persona_logits)
            persona_loss = loss_fct(persona_logits, persona_grounding)

        # extract knowledge vectors
        knowledge_sep_vectors = []
        for knowledge_sep_index_i, batch_item in zip(
            knowledge_sep_index,
            last_outputs,
        ):
            knowledge_sep_vector = batch_item[knowledge_sep_index_i]
            knowledge_sep_vectors.append(knowledge_sep_vector)

        knowledge_vector = torch.vstack(knowledge_sep_vectors)
        knowledge_logits = self.knowledge_head(knowledge_vector)

        # compute knowledge loss
        if knowledge_answer_index is not None:
            loss_fct = nn.CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits, knowledge_answer_index.view(-1))

        return BartLMV2Outputs(
            # loss
            loss=loss,
            lm_loss=lm_loss,
            persona_loss=persona_loss,
            knowledge_loss=knowledge_loss,
            # logits
            lm_logits=lm_logits,
            persona_logits=persona_logits,
            knowledge_logits=knowledge_logits,
            last_hidden_state=bart_outputs[0],
        )


class BartLMV4(BartPretrainedModel):
    def __init__(
        self,
        config: BartConfig,
        hyperparameters: BartHyperparametersV3,
        tokenizer: BartFoCusTokenizerV2,
    ) -> None:
        """
        input_ids:
            [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query><response>[dialog][-1]</response>[EOS]
        Модель у которой следующий лосс
        loss = loss_LM + loss_persona + loss_knowledge_candidates
        где
            loss_LM - лосс языковой модели
            loss_persona - лосс при классификации persona
            loss_knowledge_candidates - лосс при классификации knowledge candidates

        классификацию persona на основе:
            - <query>
            - </query>
            - [EOS]
            - [SEP] после [persona]
            - [BOS]
        классификацию knowledge_candidates на основе:
            - <query>
            - </query>
            - [EOS]
            - [SEP] после [knowledge_candidates]
            - [BOS]

        """
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
            config.d_model * 5,
            hyperparameters.persona_labels_amount,
            bias=False,
        )
        self.knowledge_candidates_head = nn.Linear(
            config.d_model * 5,
            hyperparameters.knowledge_labels_amount,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_ids_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        persona_grounding: Optional[torch.Tensor] = None,
        knowledge_answer_index: Optional[torch.Tensor] = None,
        persona_sep_index: Optional[torch.Tensor] = None,
        knowledge_candidates_sep_index: Optional[torch.Tensor] = None,
        query_eos_index: Optional[torch.Tensor] = None,
        query_bos_index: Optional[torch.Tensor] = None,
        bos_index: Optional[torch.Tensor] = None,
        eos_index: Optional[torch.Tensor] = None,
    ) -> BartLMV2Outputs:
        # assert persona_sep_index is not None
        # assert knowledge_candidates_sep_index is not None
        # assert query_eos_index is not None
        # assert query_bos_index is not None
        # assert bos_index is not None
        # assert eos_index is not None

        bart_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_outputs: torch.Tensor = bart_outputs[0]
        lm_logits: torch.Tensor = self.lm_head(last_outputs)
        loss: torch.Tensor = torch.tensor(
            0,
            device=self.device,
            dtype=torch.float,
        )
        persona_loss = None
        knowledge_loss = None
        lm_loss = None
        persona_logits = None
        knowledge_logits = None

        # compute lm loss
        if input_ids_labels is not None:
            # copy from https://github.com/pkchat-focus/FoCus/blob/main/classification_modules.py#L462 # noqa: E501
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,  # type: ignore
            )
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids_labels[..., 1:].contiguous()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss += lm_loss

        if persona_sep_index is not None:
            # extract persona vectors
            # <query></query>[EOS][SEP_persona][BOS]
            assert query_eos_index is not None
            assert query_bos_index is not None
            assert bos_index is not None
            assert eos_index is not None

            persona_feature_vectors = []
            for (
                persona_sep_index_i,
                batch_item,
                query_eos_i,
                query_bos_i,
                bos_i,
                eos_i,
            ) in zip(
                persona_sep_index,
                last_outputs,
                query_eos_index,
                query_bos_index,
                bos_index,
                eos_index,
            ):
                persona_sep_vector = batch_item[persona_sep_index_i]
                query_eos_vector = batch_item[query_eos_i]
                query_bos_vector = batch_item[query_bos_i]
                bos_vector = batch_item[bos_i]
                eos_vector = batch_item[eos_i]

                persona_feature_vector = torch.cat(
                    [
                        persona_sep_vector,
                        query_eos_vector,
                        query_bos_vector,
                        bos_vector,
                        eos_vector,
                    ],
                    dim=-1,
                )
                persona_feature_vectors.append(persona_feature_vector)

            persona_vector = torch.vstack(persona_feature_vectors)
            persona_logits = self.persona_head(persona_vector)

            # compute persona loss
            if persona_grounding is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                persona_grounding = persona_grounding.type_as(persona_logits)
                persona_loss = loss_fct(persona_logits, persona_grounding)
                loss += persona_loss

        if knowledge_candidates_sep_index is not None:
            # extract knowledge vectors
            # <query></query>[EOS][SEP_knowledge_candidates][BOS]
            assert query_eos_index is not None
            assert query_bos_index is not None
            assert bos_index is not None
            assert eos_index is not None

            knowledge_candidates_feature_vectors = []
            for (
                knowledge_sep_index_i,
                batch_item,
                query_bos_i,
                query_eos_i,
                bos_i,
                eos_i,
            ) in zip(
                knowledge_candidates_sep_index,
                last_outputs,
                query_bos_index,
                query_eos_index,
                bos_index,
                eos_index,
            ):
                knowledge_sep_vector = batch_item[knowledge_sep_index_i]
                query_eos_vector = batch_item[query_eos_i]
                query_bos_vector = batch_item[query_bos_i]
                bos_vector = batch_item[bos_i]
                eos_vector = batch_item[eos_i]
                knowledge_candidates_feature_vector = torch.cat(
                    [
                        knowledge_sep_vector,
                        query_eos_vector,
                        query_bos_vector,
                        bos_vector,
                        eos_vector,
                    ],
                    dim=-1,
                )
                knowledge_candidates_feature_vectors.append(
                    knowledge_candidates_feature_vector,
                )

            knowledge_vector = torch.vstack(knowledge_candidates_feature_vectors)
            knowledge_logits = self.knowledge_candidates_head(knowledge_vector)

            # compute knowledge loss
            if knowledge_answer_index is not None:
                loss_fct = nn.CrossEntropyLoss()
                knowledge_loss = loss_fct(
                    knowledge_logits, knowledge_answer_index.view(-1)
                )
                loss += knowledge_loss

        return BartLMV2Outputs(
            # loss
            loss=loss,
            lm_loss=lm_loss,
            persona_loss=persona_loss,
            knowledge_loss=knowledge_loss,
            # logits
            lm_logits=lm_logits,
            persona_logits=persona_logits,
            knowledge_logits=knowledge_logits,
            last_hidden_state=bart_outputs[0],
        )


class BartLMV5(BartPretrainedModel):
    def __init__(
        self,
        config: BartConfig,
        hyperparameters: BartHyperparametersV3,
        tokenizer: BartFoCusTokenizerV2,
    ) -> None:
        """
        input_ids:
            [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query><response>[dialog][-1]</response>[EOS]
        Модель у которой следующий лосс
        loss = loss_LM + loss_persona + loss_knowledge_candidates
        где
            loss_LM - лосс языковой модели
            loss_persona - лосс при классификации persona
            loss_knowledge_candidates - лосс при классификации knowledge candidates

        классификацию persona на основе:
            - <query>
            - </query>
            - [EOS]
            - [SEP] после [persona]
            - [BOS]
        классификацию knowledge_candidates на основе:
            - <query>
            - </query>
            - [EOS]
            - [SEP] после [knowledge_candidates]
            - [BOS]
        отличие отBartLMV4 в том что я буду складывать, а не конкатенировать контекстные
        вектора
        """
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
        self.knowledge_candidates_head = nn.Linear(
            config.d_model,
            hyperparameters.knowledge_labels_amount,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        persona_sep_index: Optional[torch.Tensor] = None,
        query_eos_index: Optional[torch.Tensor] = None,
        query_bos_index: Optional[torch.Tensor] = None,
        bos_index: Optional[torch.Tensor] = None,
        eos_index: Optional[torch.Tensor] = None,
        input_ids_labels: Optional[torch.Tensor] = None,
        persona_grounding: Optional[torch.Tensor] = None,
        knowledge_answer_index: Optional[torch.Tensor] = None,
        knowledge_candidates_sep_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BartLMV2Outputs:

        bart_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_outputs: torch.Tensor = bart_outputs[0]
        lm_logits: torch.Tensor = self.lm_head(last_outputs)
        loss: torch.Tensor = torch.tensor(
            0,
            device=self.device,
            dtype=torch.float,
        )
        persona_loss = None
        knowledge_loss = None
        lm_loss = None
        persona_logits = None
        knowledge_logits = None

        # compute lm loss
        if input_ids_labels is not None:
            # copy from https://github.com/pkchat-focus/FoCus/blob/main/classification_modules.py#L462 # noqa: E501
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,  # type: ignore
            )
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids_labels[..., 1:].contiguous()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss += lm_loss

        # extract persona vectors
        # <query></query>[EOS][SEP_persona][BOS]
        if persona_sep_index is not None:
            assert query_eos_index is not None
            assert query_bos_index is not None
            assert bos_index is not None
            assert eos_index is not None

            persona_feature_vectors = []
            for (
                persona_sep_index_i,
                batch_item,
                query_eos_i,
                query_bos_i,
                bos_i,
                eos_i,
            ) in zip(
                persona_sep_index,
                last_outputs,
                query_eos_index,
                query_bos_index,
                bos_index,
                eos_index,
            ):
                persona_sep_vector = batch_item[persona_sep_index_i]
                query_eos_vector = batch_item[query_eos_i]
                query_bos_vector = batch_item[query_bos_i]
                bos_vector = batch_item[bos_i]
                eos_vector = batch_item[eos_i]

                persona_sep_vector += (
                    query_eos_vector + query_bos_vector + bos_vector + eos_vector
                )
                persona_feature_vectors.append(persona_sep_vector)

            persona_vector = torch.vstack(persona_feature_vectors)
            persona_logits = self.persona_head(persona_vector)

            # compute persona loss
            if persona_grounding is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                persona_grounding = persona_grounding.type_as(persona_logits)
                persona_loss = loss_fct(persona_logits, persona_grounding)
                loss += persona_loss

        if knowledge_candidates_sep_index is not None:
            # extract knowledge vectors
            # <query></query>[EOS][SEP_knowledge_candidates][BOS]
            assert query_eos_index is not None
            assert query_bos_index is not None
            assert bos_index is not None
            assert eos_index is not None
            knowledge_candidates_feature_vectors = []
            for (
                knowledge_sep_index_i,
                batch_item,
                query_bos_i,
                query_eos_i,
                bos_i,
                eos_i,
            ) in zip(
                knowledge_candidates_sep_index,
                last_outputs,
                query_bos_index,
                query_eos_index,
                bos_index,
                eos_index,
            ):
                knowledge_sep_vector = batch_item[knowledge_sep_index_i]
                query_eos_vector = batch_item[query_eos_i]
                query_bos_vector = batch_item[query_bos_i]
                bos_vector = batch_item[bos_i]
                eos_vector = batch_item[eos_i]

                knowledge_sep_vector += (
                    query_eos_vector + query_bos_vector + bos_vector + eos_vector
                )
                knowledge_candidates_feature_vectors.append(
                    knowledge_sep_vector,
                )

            knowledge_vector = torch.vstack(knowledge_candidates_feature_vectors)
            knowledge_logits = self.knowledge_candidates_head(knowledge_vector)

            # compute knowledge loss
            if knowledge_answer_index is not None:
                loss_fct = nn.CrossEntropyLoss()
                knowledge_loss = loss_fct(
                    knowledge_logits,
                    knowledge_answer_index.view(-1),
                )
                loss += knowledge_loss

        return BartLMV2Outputs(
            # loss
            loss=loss,
            lm_loss=lm_loss,
            persona_loss=persona_loss,
            knowledge_loss=knowledge_loss,
            # logits
            lm_logits=lm_logits,
            persona_logits=persona_logits,
            knowledge_logits=knowledge_logits,
            last_hidden_state=bart_outputs[0],
        )
