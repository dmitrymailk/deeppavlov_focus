import os
from typing import List, Optional, TypedDict
from typing import cast

from core.dataloaders.focus.focus_dataloader import FoCusDatasetV1
from core.dataloaders.focus.models.bart_dataloaders import (
    BartFoCusDatasetSampleDictV3,
    BartFoCusDatasetSampleDictV4,
    BartFoCusDatasetSampleDictV5,
    BartFoCusDatasetSampleV5,
    PytorchFoCusDatasetV3,
    PytorchFoCusDatasetV4,
)
from core.hyperparameters.bart_hyperparameters import BartHyperparametersV3
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV2

from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader

from core.utils import PytorchDatasetFactory


class FoCusLightningDataModuleV2DictV1(TypedDict):
    input_ids: torch.Tensor
    input_ids_labels: torch.Tensor
    attention_mask: torch.Tensor
    persona_grounding: torch.Tensor
    knowledge_answer_index: torch.Tensor
    persona_sep_index: torch.Tensor
    knowledge_sep_index: torch.Tensor
    dialog_bos_index: torch.Tensor
    dialog_eos_index: torch.Tensor


class FoCusLightningDataModuleV3DictV1(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    persona_grounding: torch.Tensor
    knowledge_answer_index: torch.Tensor
    persona_sep_index: torch.Tensor
    knowledge_candidates_sep_index: torch.Tensor
    query_eos_index: torch.Tensor
    query_bos_index: torch.Tensor
    bos_index: torch.Tensor
    eos_index: torch.Tensor


class FoCusLightningDataModuleV5DictV1(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    labels: torch.Tensor
    persona_grounding: torch.Tensor
    knowledge_candidates_answer_index: torch.Tensor

    persona_ids: torch.Tensor
    persona_attention_mask: torch.Tensor

    knowledge_ids: torch.Tensor
    knowledge_attention_mask: torch.Tensor

    knowledge_candidates_ids: torch.Tensor
    knowledge_candidates_attention_mask: torch.Tensor

    query_ids: torch.Tensor
    query_attention_mask: torch.Tensor


class FoCusLightningDataModuleV4(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: BartHyperparametersV3,
        tokenizer: BartFoCusTokenizerV2,
        debug_status: int = 0,
    ) -> None:
        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.debug_status = debug_status

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetV1(input_dataset_path=self.train_path_dataset)
        valid_dataset = FoCusDatasetV1(input_dataset_path=self.valid_path_dataset)

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = PytorchFoCusDatasetV4(
            dataset=train_dataset,  # type: ignore
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
        )
        self.valid_dataset = PytorchFoCusDatasetV4(
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
        batch: List[BartFoCusDatasetSampleDictV4],
    ) -> FoCusLightningDataModuleV3DictV1:
        max_input_ids_len = max([len(item["input_ids"]) for item in batch])
        max_labels_len = max([len(item["labels"]) for item in batch])

        batch_input_ids = []
        batch_attention_mask = []
        batch_persona_grounding = []
        batch_knowledge_candidates_answer_index = []
        batch_knowledge_candidates_sep_index = []
        batch_persona_sep_index = []
        batch_query_bos_index = []
        batch_query_eos_index = []
        batch_bos_index = []
        batch_eos_index = []
        batch_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            persona_grounding = item["persona_grounding"]
            knowledge_answer_index = item["knowledge_candidates_answer_index"]
            persona_sep_index = item["persona_sep_index"]
            knowledge_candidates_sep_index = item["knowledge_candidates_sep_index"]
            query_bos_index = item["query_bos_index"]
            query_eos_index = item["query_eos_index"]
            bos_index = item["bos_index"]
            eos_index = item["eos_index"]
            labels = item["labels"]

            pad_tokens = cast(
                List[int],
                [self.tokenizer.pad_token_id] * (max_input_ids_len - len(input_ids)),
            )
            pad_attention = [0] * (max_input_ids_len - len(attention_mask))

            pad_labels = cast(
                List[int],
                [self.tokenizer.pad_token_id] * (max_labels_len - len(item["labels"])),
            )

            input_ids.extend(pad_tokens)
            attention_mask.extend(pad_attention)

            labels.extend(pad_labels)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_persona_grounding.append(persona_grounding)
            batch_knowledge_candidates_answer_index.append([knowledge_answer_index])
            batch_persona_sep_index.append([persona_sep_index])
            batch_knowledge_candidates_sep_index.append(
                [knowledge_candidates_sep_index],
            )
            batch_query_bos_index.append([query_bos_index])
            batch_query_eos_index.append([query_eos_index])
            batch_bos_index.append([bos_index])
            batch_eos_index.append([eos_index])
            batch_labels.append(labels)

        return FoCusLightningDataModuleV3DictV1(
            input_ids=torch.tensor(batch_input_ids),
            labels=torch.tensor(batch_labels),
            attention_mask=torch.tensor(batch_attention_mask),
            persona_grounding=torch.tensor(batch_persona_grounding),
            knowledge_answer_index=torch.tensor(
                batch_knowledge_candidates_answer_index,
            ),
            persona_sep_index=torch.tensor(batch_persona_sep_index),
            knowledge_candidates_sep_index=torch.tensor(
                batch_knowledge_candidates_sep_index,
            ),
            query_eos_index=torch.tensor(batch_query_eos_index),
            query_bos_index=torch.tensor(batch_query_bos_index),
            bos_index=torch.tensor(batch_bos_index),
            eos_index=torch.tensor(batch_eos_index),
        )


class FoCusLightningDataModuleV3(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: BartHyperparametersV3,
        tokenizer: BartFoCusTokenizerV2,
        debug_status: int = 0,
    ) -> None:
        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.debug_status = debug_status

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetV1(input_dataset_path=self.train_path_dataset)
        valid_dataset = FoCusDatasetV1(input_dataset_path=self.valid_path_dataset)

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = PytorchFoCusDatasetV3(
            dataset=train_dataset,  # type: ignore
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
        )
        self.valid_dataset = PytorchFoCusDatasetV3(
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
        batch: List[BartFoCusDatasetSampleDictV3],
    ) -> FoCusLightningDataModuleV3DictV1:
        max_input_ids_len = max([len(item["input_ids"]) for item in batch])
        max_labels_len = max([len(item["labels"]) for item in batch])

        batch_input_ids = []
        batch_attention_mask = []
        batch_persona_grounding = []
        batch_knowledge_candidates_answer_index = []
        batch_knowledge_candidates_sep_index = []
        batch_persona_sep_index = []
        batch_query_bos_index = []
        batch_query_eos_index = []
        batch_bos_index = []
        batch_eos_index = []
        batch_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            persona_grounding = item["persona_grounding"]
            knowledge_answer_index = item["knowledge_candidates_answer_index"]
            persona_sep_index = item["persona_sep_index"]
            knowledge_candidates_sep_index = item["knowledge_candidates_sep_index"]
            query_bos_index = item["query_bos_index"]
            query_eos_index = item["query_eos_index"]
            bos_index = item["bos_index"]
            eos_index = item["eos_index"]
            labels = item["labels"]

            pad_tokens = cast(
                List[int],
                [self.tokenizer.pad_token_id] * (max_input_ids_len - len(input_ids)),
            )
            pad_attention = [0] * (max_input_ids_len - len(attention_mask))

            pad_labels = cast(
                List[int],
                [self.tokenizer.pad_token_id] * (max_labels_len - len(item["labels"])),
            )

            input_ids.extend(pad_tokens)
            attention_mask.extend(pad_attention)

            labels.extend(pad_labels)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_persona_grounding.append(persona_grounding)
            batch_knowledge_candidates_answer_index.append([knowledge_answer_index])
            batch_persona_sep_index.append([persona_sep_index])
            batch_knowledge_candidates_sep_index.append(
                [knowledge_candidates_sep_index],
            )
            batch_query_bos_index.append([query_bos_index])
            batch_query_eos_index.append([query_eos_index])
            batch_bos_index.append([bos_index])
            batch_eos_index.append([eos_index])
            batch_labels.append(labels)

        return FoCusLightningDataModuleV3DictV1(
            input_ids=torch.tensor(batch_input_ids),
            labels=torch.tensor(batch_labels),
            attention_mask=torch.tensor(batch_attention_mask),
            persona_grounding=torch.tensor(batch_persona_grounding),
            knowledge_answer_index=torch.tensor(
                batch_knowledge_candidates_answer_index,
            ),
            persona_sep_index=torch.tensor(batch_persona_sep_index),
            knowledge_candidates_sep_index=torch.tensor(
                batch_knowledge_candidates_sep_index,
            ),
            query_eos_index=torch.tensor(batch_query_eos_index),
            query_bos_index=torch.tensor(batch_query_bos_index),
            bos_index=torch.tensor(batch_bos_index),
            eos_index=torch.tensor(batch_eos_index),
        )


class FoCusLightningDataModuleV5(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: BartHyperparametersV3,
        tokenizer: BartFoCusTokenizerV2,
        debug_status: int = 0,
    ) -> None:
        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.debug_status = debug_status

    def setup(self, stage: Optional[str] = None):
        train_dataset = FoCusDatasetV1(input_dataset_path=self.train_path_dataset)
        valid_dataset = FoCusDatasetV1(input_dataset_path=self.valid_path_dataset)

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]  # type: ignore
            valid_dataset = valid_dataset[:2]  # type: ignore
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]  # type: ignore
            valid_dataset = valid_dataset  # type: ignore

        self.train_dataset = PytorchDatasetFactory(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            dataset_sample_class=BartFoCusDatasetSampleV5,
            hyperparameters=self.hyperparameters,
        )

        self.valid_dataset = PytorchDatasetFactory(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            dataset_sample_class=BartFoCusDatasetSampleV5,
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

    def pad_to_max(self, array, max_len, pad_token_id):
        return array + [pad_token_id] * (max_len - len(array))

    def collate_fn(
        self,
        batch: List[BartFoCusDatasetSampleDictV5],
    ) -> FoCusLightningDataModuleV5DictV1:
        label_pad_keys = [
            "input_ids",
            "labels",
            "persona_ids",
            "knowledge_ids",
            "knowledge_candidates_ids",
            "query_ids",
        ]

        attention_pad_keys = [
            "attention_mask",
            "persona_attention_mask",
            "knowledge_attention_mask",
            "knowledge_candidates_attention_mask",
            "query_attention_mask",
        ]

        label_max_lens = []
        attention_max_lens = []

        for pad_key in label_pad_keys:
            max_len = max([len(item[pad_key]) for item in batch])
            label_max_lens.append(max_len)

        for pad_key in attention_pad_keys:
            max_len = max([len(item[pad_key]) for item in batch])
            attention_max_lens.append(max_len)

        # padd all the arrays
        for i, pad_key in enumerate(label_pad_keys):
            max_len = label_max_lens[i]
            for item in batch:
                item[pad_key] = self.pad_to_max(
                    item[pad_key],
                    max_len,
                    self.tokenizer.pad_token_id,
                )

        for i, pad_key in enumerate(attention_pad_keys):
            max_len = attention_max_lens[i]
            for item in batch:
                item[pad_key] = self.pad_to_max(
                    item[pad_key],
                    max_len,
                    0,
                )

        batch_keys = batch[0].keys()
        batch_dict = {key: [] for key in batch_keys}

        for item in batch:
            for key in batch_keys:
                batch_dict[key].append(item[key])

        # convert to tensors
        for key in batch_keys:
            batch_dict[key] = torch.tensor(batch_dict[key])  # type: ignore

        return FoCusLightningDataModuleV5DictV1(
            **batch_dict,  # type: ignore
        )
