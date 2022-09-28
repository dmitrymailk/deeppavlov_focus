from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput  # type: ignore


class BartOutputV1(ModelOutput):
    # Seq2SeqLMOutput
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # custom fields
    lm_loss: Optional[torch.Tensor]
    persona_loss: Optional[torch.Tensor]
    knowledge_loss: Optional[torch.Tensor]
    persona_logits: torch.Tensor
    knowledge_logits: torch.Tensor
    last_hidden_state: torch.Tensor
