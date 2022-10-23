from typing import Optional, Tuple, Union

from transformers import (
    MPNetModel,  # type: ignore
    MPNetForSequenceClassification,  # type: ignore
)
from transformers.models.mpnet.modeling_mpnet import (
    MPNetClassificationHead,
    SequenceClassifierOutput,  # type: ignore
)

import torch
from torch import nn


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
