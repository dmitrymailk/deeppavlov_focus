from typing import List, TypedDict

from torch.utils.data import Dataset

from core.dataloaders.focus.focus_dataloader import (
    FoCusDatasetKnowledgeSampleDictV1,
    FoCusDatasetKnowledgeV1,
    FoCusDatasetKnowledgeV2,
)
from core.hyperparameters.debertav3_hyperparameters import DebertaV3HyperparametersV1

from transformers.utils.dummy_sentencepiece_objects import DebertaV2Tokenizer


class DebertaV3FoCusKnowledgeDatasetSampleDictV1(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    labels: int
    unique_id: str


class DebertaV3FoCusKnowledgeDatasetSampleV1:
    def __init__(
        self,
        dataset_sample: FoCusDatasetKnowledgeSampleDictV1,
        tokenizer: DebertaV2Tokenizer,
        h_params: DebertaV3HyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: DebertaV2Tokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> DebertaV3FoCusKnowledgeDatasetSampleDictV1:
        """
        Returns:
            input_ids (List[int]):
                [BOS][knowledge_candidate][dialog][-2][EOS]
                [dialog][-2] - это последний вопрос от пользователя
                [knowledge_candidate] - это кандидат на знание. большая часть
                    будет неправильными ответами
            labels (int): 0 или 1. ложь или правда. использовалось ли знание для ответа
                или нет.
        """
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens

        knowledge_candidate = self.dataset_sample["knowledge_candidate"]
        dialog = self.dataset_sample["dialog"]
        knowledge_candidate_usage = self.dataset_sample["knowledge_candidate_usage"]
        unique_id = self.dataset_sample["unique_id"]

        encoded_knowledge_candidate = self.tokenizer.encode(  # type: ignore
            knowledge_candidate,
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_candidates_tokens,
        )

        encoded_dialog = self.tokenizer.encode(  # type: ignore
            dialog[-2],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        input_ids = [
            self.bos_token_id,
            *encoded_knowledge_candidate,
            *encoded_dialog,
            self.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return DebertaV3FoCusKnowledgeDatasetSampleDictV1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=knowledge_candidate_usage,
            unique_id=unique_id,
        )


class DebertaV3PytorchFoCusKnowledgeDatasetV1(Dataset):
    def __init__(
        self,
        dataset: FoCusDatasetKnowledgeV1 | FoCusDatasetKnowledgeV2,
        tokenizer: DebertaV2Tokenizer,
        hyperparameters: DebertaV3HyperparametersV1,
    ) -> None:
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DebertaV3FoCusKnowledgeDatasetSampleDictV1:
        dataset_sample = self.dataset[index]
        train_sample = DebertaV3FoCusKnowledgeDatasetSampleV1(
            dataset_sample=dataset_sample,
            tokenizer=self.tokenizer,
            h_params=self.hyperparameters,
        )
        train_sample = train_sample.get_dict()
        return train_sample
