from typing import List

from core.dataloaders.focus.lighting.base_lighting import LightningDataModuleV1
from core.dataloaders.focus.models.mpnet_dataloaders import (
    MPNetV3FoCusKnowledgeDatasetSampleDictV1,
)

import torch


class MPNetLightingDataModuleV1(LightningDataModuleV1):
    def train_collate_fn(
        self,
        batch: List[MPNetV3FoCusKnowledgeDatasetSampleDictV1],
    ):
        pad_id = self.tokenizer.pad_token_id  # type: ignore
        batch_sent_1 = [sample["sentence_1"] for sample in batch]
        batch_sent_2 = [sample["sentence_2"] for sample in batch]
        attention_mask_1 = [sample["attention_mask_1"] for sample in batch]
        attention_mask_2 = [sample["attention_mask_2"] for sample in batch]

        max_len_1 = self._get_max_len(batch_sent_1)
        max_len_2 = self._get_max_len(batch_sent_2)

        batch_sent_1 = self._padding(batch_sent_1, pad_id, max_len=max_len_1)
        batch_sent_2 = self._padding(batch_sent_2, pad_id, max_len=max_len_2)
        batch_attention_mask_1 = self._padding(attention_mask_1, 0, max_len=max_len_1)
        batch_attention_mask_2 = self._padding(attention_mask_2, 0, max_len=max_len_2)
        batch_utterance_ids = [sample["utterance_id"] for sample in batch]
        batch_scores = [sample["score"] for sample in batch]

        return {
            "sentence_1": torch.tensor(batch_sent_1),
            "sentence_2": torch.tensor(batch_sent_2),
            "attention_mask_1": torch.tensor(batch_attention_mask_1),
            "attention_mask_2": torch.tensor(batch_attention_mask_2),
            "utterance_id": batch_utterance_ids,
            "score": batch_scores,
        }
