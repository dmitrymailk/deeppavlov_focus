import os
from typing import List, Optional, TypedDict
from typing import cast

from core.dataloaders.focus.focus_dataloader import (
    FoCusDatasetKnowledgeV1,
    FoCusDatasetKnowledgeV2,
)
from core.dataloaders.focus.models.debertav3_dataloaders import (
    DebertaV3FoCusKnowledgeDatasetSampleDictV1,
    DebertaV3FoCusKnowledgeDatasetSampleV1,
    DebertaV3FoCusKnowledgeDatasetSampleV2,
    DebertaV3FoCusKnowledgeDatasetSampleV3,
    DebertaV3FoCusKnowledgeDatasetSampleV4,
    DebertaV3PytorchFoCusKnowledgeDatasetV1,
)
from core.hyperparameters.debertav3_hyperparameters import DebertaV3HyperparametersV1

from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader

from transformers.utils.dummy_sentencepiece_objects import DebertaV2Tokenizer

from core.utils import PytorchDatasetFactory


class DebertaV3FoCusLightningDataModuleV1DictV1(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    unique_ids: List[str]


class DebertaV3FoCusLightningDataModuleV1(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        debug_status: int = 0,
    ) -> None:
        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.debug_status = debug_status

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetKnowledgeV1(
            input_dataset_path=self.train_path_dataset,
        )
        valid_dataset = FoCusDatasetKnowledgeV1(
            input_dataset_path=self.valid_path_dataset,
        )

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = DebertaV3PytorchFoCusKnowledgeDatasetV1(
            dataset=train_dataset,  # type: ignore
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
        )
        self.valid_dataset = DebertaV3PytorchFoCusKnowledgeDatasetV1(
            dataset=valid_dataset,  # type: ignore
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.hyperparameters.train_batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),  # type: ignore
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,  # type: ignore
            batch_size=self.hyperparameters.valid_batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),  # type: ignore
            collate_fn=self.collate_fn,
        )

    def collate_fn(
        self,
        batch: List[DebertaV3FoCusKnowledgeDatasetSampleDictV1],
    ) -> DebertaV3FoCusLightningDataModuleV1DictV1:
        max_input_ids_len = max([len(item["input_ids"]) for item in batch])

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_unique_ids = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            unique_id = item["unique_id"]

            pad_tokens = cast(
                List[int],
                [self.tokenizer.pad_token_id]  # type: ignore
                * (max_input_ids_len - len(input_ids)),
            )
            pad_attention = [0] * (max_input_ids_len - len(attention_mask))

            input_ids.extend(pad_tokens)
            attention_mask.extend(pad_attention)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_unique_ids.append(unique_id)

        return DebertaV3FoCusLightningDataModuleV1DictV1(
            input_ids=torch.tensor(batch_input_ids),
            labels=torch.tensor(batch_labels),
            attention_mask=torch.tensor(batch_attention_mask),
            unique_ids=batch_unique_ids,
        )


class DebertaV3FoCusLightningDataModuleV2(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        debug_status: int = 0,
    ) -> None:
        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.debug_status = debug_status

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.train_path_dataset,
            is_train=True,
        )
        valid_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.valid_path_dataset,
            is_train=False,
        )

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore
        # DebertaV3FoCusKnowledgeDatasetSampleV1
        self.train_dataset = PytorchDatasetFactory(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV1,
        )

        self.valid_dataset = PytorchDatasetFactory(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV1,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.hyperparameters.train_batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),  # type: ignore
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,  # type: ignore
            batch_size=self.hyperparameters.valid_batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),  # type: ignore
            collate_fn=self.collate_fn,
        )

    def collate_fn(
        self,
        batch: List[DebertaV3FoCusKnowledgeDatasetSampleDictV1],
    ) -> DebertaV3FoCusLightningDataModuleV1DictV1:
        max_input_ids_len = max([len(item["input_ids"]) for item in batch])

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_unique_ids = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            unique_id = item["unique_id"]

            pad_tokens = cast(
                List[int],
                [self.tokenizer.pad_token_id]  # type: ignore
                * (max_input_ids_len - len(input_ids)),
            )
            pad_attention = [0] * (max_input_ids_len - len(attention_mask))

            input_ids.extend(pad_tokens)
            attention_mask.extend(pad_attention)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_unique_ids.append(unique_id)

        return DebertaV3FoCusLightningDataModuleV1DictV1(
            input_ids=torch.tensor(batch_input_ids),
            labels=torch.tensor(batch_labels),
            attention_mask=torch.tensor(batch_attention_mask),
            unique_ids=batch_unique_ids,
        )


class DebertaV3FoCusLightningDataModuleV3(DebertaV3FoCusLightningDataModuleV2):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        debug_status: int = 0,
    ) -> None:
        super().__init__(
            train_path_dataset,
            valid_path_dataset,
            hyperparameters,
            tokenizer,
            debug_status,
        )

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.train_path_dataset,
            is_train=True,
        )
        valid_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.valid_path_dataset,
            is_train=False,
        )

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = PytorchDatasetFactory(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV2,
        )
        self.valid_dataset = PytorchDatasetFactory(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV2,
        )


class DebertaV3FoCusLightningDataModuleV4(DebertaV3FoCusLightningDataModuleV2):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        debug_status: int = 0,
    ) -> None:
        super().__init__(
            train_path_dataset,
            valid_path_dataset,
            hyperparameters,
            tokenizer,
            debug_status,
        )

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.train_path_dataset,
            is_train=True,
        )
        valid_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.valid_path_dataset,
            is_train=False,
        )

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = PytorchDatasetFactory(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV3,
        )
        self.valid_dataset = PytorchDatasetFactory(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV3,
        )


class DebertaV3FoCusLightningDataModuleV5(DebertaV3FoCusLightningDataModuleV2):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        debug_status: int = 0,
    ) -> None:
        super().__init__(
            train_path_dataset,
            valid_path_dataset,
            hyperparameters,
            tokenizer,
            debug_status,
        )

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.train_path_dataset,
            is_train=True,
        )
        valid_dataset = FoCusDatasetKnowledgeV2(
            input_dataset_path=self.valid_path_dataset,
            is_train=False,
        )

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = PytorchDatasetFactory(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV4,
        )
        self.valid_dataset = PytorchDatasetFactory(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV4,
        )


class DebertaV3FoCusLightningDataModuleV6(DebertaV3FoCusLightningDataModuleV2):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        debug_status: int = 0,
    ) -> None:
        super().__init__(
            train_path_dataset,
            valid_path_dataset,
            hyperparameters,
            tokenizer,
            debug_status,
        )

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetKnowledgeV1(
            input_dataset_path=self.train_path_dataset,
        )
        valid_dataset = FoCusDatasetKnowledgeV1(
            input_dataset_path=self.valid_path_dataset,
        )

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = PytorchDatasetFactory(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV4,
        )
        self.valid_dataset = PytorchDatasetFactory(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=DebertaV3FoCusKnowledgeDatasetSampleV4,
        )
