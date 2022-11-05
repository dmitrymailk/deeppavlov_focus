from typing import List, Optional

from core.base_models.model_outputs.debertav3_outputs import DebertaV3OutputV1

import torch
from torch import nn

from transformers import DebertaV2Config  # type: ignore
from transformers import DebertaV2ForSequenceClassification  # type: ignore
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler,
    DebertaV2Model,
    StableDropout,
)

from kornia.losses import FocalLoss


class DebertaV3ForClassificationV1(DebertaV2ForSequenceClassification):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)

        num_labels = 2
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unique_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return DebertaV3OutputV1(
            loss=loss,
            logits=logits,
            unique_ids=unique_ids,
        )


class DebertaV3ForClassificationV2(DebertaV2ForSequenceClassification):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)

        num_labels = 2
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unique_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0][:, 0]
        logits = self.classifier(encoder_layer)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return DebertaV3OutputV1(
            loss=loss,
            logits=logits,
            unique_ids=unique_ids,
        )


class DebertaV3PersonaClassificationV1(DebertaV2ForSequenceClassification):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)

        num_labels = 2
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unique_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # sentences_outpus = []
        predicted_logits = []
        loss = torch.tensor(0.0, device=self.device)
        for i, batch_input_ids, batch_attention_mask in zip(
            range(len(input_ids)), input_ids, attention_mask
        ):
            outputs = self.deberta(
                batch_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=batch_attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            logits = self.classifier(pooled_output)
            predicted_logits.append(logits)

            if labels is not None:
                batch_labels = labels[i]
                loss_fct = nn.CrossEntropyLoss()
                loss += loss_fct(
                    logits,
                    batch_labels,
                )
        stacked_logits = torch.stack(predicted_logits, dim=0)
        return DebertaV3OutputV1(
            loss=loss,
            logits=stacked_logits,
            unique_ids=unique_ids,
        )


class DebertaV3PersonaClassificationV2(DebertaV2ForSequenceClassification):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)

        num_labels = 1
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unique_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        encoder_layer = self.pooler(encoder_layer)
        logits = self.classifier(encoder_layer)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels)

        return DebertaV3OutputV1(
            loss=loss,
            logits=logits,
            unique_ids=unique_ids,
        )


class DebertaV3PersonaClassificationV3(DebertaV2ForSequenceClassification):
    def __init__(
        self,
        config: DebertaV2Config,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__(config)

        num_labels = 2
        self.num_labels = num_labels
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.loss_fcn = FocalLoss(
            gamma=2.0,
            alpha=0.5,
            reduction="mean",
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unique_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_ids = input_ids.to(self.device)  # type: ignore
        attention_mask = attention_mask.to(self.device)  # type: ignore
        if labels is not None:
            labels = labels.to(self.device)  # type: ignore

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        encoder_layer = self.pooler(encoder_layer)
        logits = self.classifier(encoder_layer)

        loss = torch.tensor(0.0).to(self.device)
        if labels is not None:
            # assert self.class_weights is not None
            # loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
            loss = self.loss_fcn(logits, labels)

        return DebertaV3OutputV1(
            loss=loss,
            logits=logits,
        )


class DebertaV3ForSentenceEmbeddingV1(DebertaV2ForSequenceClassification):
    def __init__(
        self,
        config: DebertaV2Config,
        normalize: bool = True,
    ):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.normalize = normalize
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unique_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_ids = input_ids.to(self.device)  # type: ignore
        attention_mask = attention_mask.to(self.device)  # type: ignore

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        embeddings = self.mean_pooling(encoder_layer, attention_mask)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, embeddings, attention_mask):
        # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        )
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9,
        )
