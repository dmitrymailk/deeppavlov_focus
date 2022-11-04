import os
from typing import Optional, Any, Callable


from core.utils import PytorchDatasetFactory

from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

from transformers import AutoTokenizer  # type: ignore


class LightningDataModuleV1(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str,
        valid_path_dataset: str,
        hyperparameters: Any,
        tokenizer: AutoTokenizer,
        base_train_dataset_class: Any = None,
        base_valid_dataset_class: Any = None,
        base_train_sample_class: Any = None,
        base_valid_sample_class: Any = None,
        debug_status: int = 0,
    ) -> None:
        assert debug_status in [0, 1, 2]
        assert base_train_dataset_class is not None
        assert base_valid_dataset_class is not None
        assert base_train_sample_class is not None
        assert base_valid_sample_class is not None

        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.base_train_dataset_class = base_train_dataset_class
        self.base_valid_dataset_class = base_valid_dataset_class
        self.base_train_sample_class = base_train_sample_class
        self.base_valid_sample_class = base_valid_sample_class

        self.debug_status = debug_status

    def setup(self, stage: Optional[str] = None):
        train_dataset = self.base_train_dataset_class(
            input_dataset_path=self.train_path_dataset,
        )
        valid_dataset = self.base_valid_dataset_class(
            input_dataset_path=self.valid_path_dataset,
        )

        if self.debug_status == 1:
            train_dataset = train_dataset[:2]
            valid_dataset = valid_dataset[:2]
        elif self.debug_status == 2:
            train_dataset = train_dataset[:15000]
            valid_dataset = valid_dataset

        self.train_dataset = PytorchDatasetFactory(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=self.base_train_sample_class,
        )

        self.valid_dataset = PytorchDatasetFactory(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample_class=self.base_valid_sample_class,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.hyperparameters.train_batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),  # type: ignore
            collate_fn=self.train_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,  # type: ignore
            batch_size=self.hyperparameters.valid_batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),  # type: ignore
            collate_fn=self.valid_collate_fn,
        )

    def train_collate_fn(self, batch):
        raise NotImplementedError

    def valid_collate_fn(self, batch):
        return self.train_collate_fn(batch)

    def _padding(self, list_of_tokens: list, pad_id: int, max_len: int) -> list:
        return [
            tokens + [pad_id] * (max_len - len(tokens)) for tokens in list_of_tokens
        ]

    def _get_max_len(self, list_of_tokens: list) -> int:
        return max([len(tokens) for tokens in list_of_tokens])
