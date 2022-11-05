from typing import Optional, Tuple, Union

from transformers import (
    MPNetModel,  # type: ignore
    MPNetForSequenceClassification,  # type: ignore
)
from transformers.models.mpnet.modeling_mpnet import (
    MPNetClassificationHead,
    SequenceClassifierOutput,  # type: ignore
)

from kornia.losses import FocalLoss

import torch
from torch import nn
import torch.nn.functional as F


class FocalLossV1(nn.CrossEntropyLoss):
    """Focal loss for classification tasks on imbalanced datasets"""

    # https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
    def __init__(
        self,
        gamma,
        alpha=None,
        ignore_index=-100,
    ):
        super().__init__(
            weight=alpha,
            ignore_index=ignore_index,
            reduction="mean",
        )
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss)


class MPNetForSequenceClassificationV1(MPNetForSequenceClassification):
    def __init__(self, config, cross_entropy_loss_weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mpnet = MPNetModel(config)
        self.classifier = MPNetClassificationHead(config)
        self.cross_entropy_loss_weights = cross_entropy_loss_weights

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = None
            if self.cross_entropy_loss_weights is None:
                loss_fct = nn.CrossEntropyLoss()
            else:
                loss_fct = nn.CrossEntropyLoss(
                    weight=torch.tensor(
                        self.cross_entropy_loss_weights,
                        dtype=torch.float32,
                        device=labels.device,
                    ),
                )

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetForSequenceClassificationV2(MPNetForSequenceClassification):
    def __init__(self, config, cross_entropy_loss_weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mpnet = MPNetModel(config)
        self.classifier = MPNetClassificationHead(config)
        self.cross_entropy_loss_weights = cross_entropy_loss_weights

        # self.loss_fcn = FocalLossV1(
        #     gamma=2.0,
        #     alpha=cross_entropy_loss_weights,
        # )
        self.loss_fcn = FocalLoss(
            gamma=2.0,
            alpha=0.5,
            reduction="mean",
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fcn(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return SequenceClassifierOutput(
            loss=loss,  # type: ignore
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetForSentenceEmbeddingV1(MPNetForSequenceClassification):
    def __init__(self, config, normalize=True):
        super().__init__(config)

        self.mpnet = MPNetModel(config)
        self.normalize = normalize

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        model_output = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embeddings = self.mean_pooling(model_output[0], attention_mask)
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


class MPNetForSentenceEmbeddingV2(MPNetForSequenceClassification):
    def __init__(self, config, normalize=True):
        super().__init__(config)

        self.mpnet = MPNetModel(config)
        self.freeze_mpnet = MPNetModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
        )
        for param in self.freeze_mpnet.parameters():  # type: ignore
            param.requires_grad = False
        self.normalize = normalize

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # model_output = self.mpnet(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # embeddings = self.mean_pooling(model_output[0], attention_mask)
        # embeddings2 = None

        embeddings2 = self.freeze_mpnet(  # type: ignore
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embeddings2 = self.mean_pooling(
            embeddings2[0],
            attention_mask=attention_mask,
        )

        if self.normalize:
            embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

        return embeddings2

    def mean_pooling(self, embeddings, attention_mask):
        # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        )
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9,
        )
