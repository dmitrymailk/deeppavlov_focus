from typing import List, Optional

import torch

from transformers.utils import ModelOutput  # type: ignore


class DebertaV3OutputV1(ModelOutput):
    # Seq2SeqLMOutput
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor
    # они нужны чтобы потом посчитать accuracy
    # по изначальному датасету с knowledge_candidates
    unique_ids: List[int]


class DebertaV3OutputV2(ModelOutput):
    # Seq2SeqLMOutput
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor
