import json
import os
from itertools import chain
from typing import Dict, List, TypedDict

from core.hyperparameters.bart_hyperparameters import BartHyperparametersV1
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1
from core.utils import FoCusTfIdf

from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader, Dataset


class FoCusDatasetSampleDictV1(TypedDict):
    persona: List[str]
    knowledge_candidates: List[str]
    persona_grounding: List[int]
    dialog: List[int]
    knowledge_answer_index: int
    knowledge: List[str]


class FoCusDatasetSampleV1:
    __slots__ = (
        "persona",
        "knowledge_candidates",
        "persona_grounding",
        "dialog",
        "knowledge_answer_index",
        "knowledge",
    )

    def __init__(
        self,
        persona: List[str],
        knowledge_candidates: List[str],
        persona_grounding: List[int],
        dialog: List[str],
        knowledge: List[str],
        knowledge_answer_index: int,
    ) -> None:
        self.persona = persona
        self.knowledge_candidates = knowledge_candidates
        self.persona_grounding = persona_grounding
        self.knowledge_answer_index = knowledge_answer_index
        self.dialog = dialog
        self.knowledge = knowledge

    def get_dict(self) -> FoCusDatasetSampleDictV1:
        return {
            "persona": self.persona,
            "knowledge_candidates": self.knowledge_candidates,
            "persona_grounding": self.persona_grounding,
            "dialog": self.dialog,
            "knowledge_answer_index": self.knowledge_answer_index,
            "knowledge": self.knowledge,
        }


class BartFoCusDatasetSampleDictV1(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]


class BartFoCusDatasetSampleV1:
    # fmt: off
    """
        [BOS][persona][SEP][knowledge][SEP][dialog][:-1][SEP]<dialog>[dialog][-1]</dialog>
        - [dialog] - набор диалоговых пар
        - persona - все предложения персоны
        - knowledge - топ наиболее похожих предложений из knowledge к контексту диалога
        - [dialog][:-1] - все диалоговые пары, кроме ответа бота
        - <dialog>[dialog][-1]</dialog> - ответ бота
    """
    # fmt: on

    def __init__(
        self,
        focus_dataset_sample: FoCusDatasetSampleDictV1 = None,
        tokenizer: BartFoCusTokenizerV1 = None,
        h_params: BartHyperparametersV1 = None,
    ) -> None:
        self.focus_dataset_sample = focus_dataset_sample
        self.tokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

        self.dialog_bos = self.__get_token_id(h_params.dialog_bos_token)
        self.dialog_eos = self.__get_token_id(h_params.dialog_eos_token)

    def __get_token_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    def __flat_list(self, list_of_lists: List[List]) -> List:
        return list(chain.from_iterable(list_of_lists))

    def get_dict(self) -> BartFoCusDatasetSampleDictV1:
        dialog_history_length = self.h_params.dialog_history_length
        context_length = self.h_params.context_length

        persona = self.focus_dataset_sample["persona"]
        dialog = self.focus_dataset_sample["dialog"]
        knowledge = self.focus_dataset_sample["knowledge"]

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
        )

        dialog_history = dialog[-2 * dialog_history_length :]
        dialog_history_feature = self.tokenizer.batch_encode_plus(
            dialog_history[:-1],
            add_special_tokens=False,
            truncation=True,
        )
        dialog_history_target = self.tokenizer.batch_encode_plus(
            dialog_history[-1:],
            add_special_tokens=False,
            truncation=True,
        )

        # контекст на основе которого подбирается knowledge
        query_context = dialog_history_feature["input_ids"][-context_length:]
        encoded_knowledge = self.tokenizer.batch_encode_plus(
            knowledge,
            add_special_tokens=False,
            truncation=True,
        )

        tf_idf = FoCusTfIdf(corpus=encoded_knowledge["input_ids"])
        most_similar_knowledge = tf_idf.top_similar(
            query=query_context,
        )

        # [BOS][persona][SEP][knowledge][SEP][dialog][:-1][SEP]<dialog>[dialog][-1]</dialog>
        flat_persona = self.__flat_list(encoded_persona["input_ids"])
        flat_knowledge = self.__flat_list(most_similar_knowledge)
        flat_dialog_history = self.__flat_list(dialog_history_feature["input_ids"])
        flat_bot_response = self.__flat_list(dialog_history_target["input_ids"])

        flat_persona = flat_persona[: self.h_params.max_persona_tokens]
        flat_knowledge = flat_knowledge[: self.h_params.max_knowledge_tokens]
        flat_dialog_history = flat_dialog_history[
            : self.h_params.max_dialog_history_tokens
        ]
        flat_bot_response = flat_bot_response[: self.h_params.max_bot_response_tokens]

        input_sequence = [
            self.bos_token_id,
            *flat_persona,
            self.sep_token_id,
            *flat_knowledge,
            self.sep_token_id,
            *flat_dialog_history,
            self.sep_token_id,
            self.dialog_bos,
            *flat_bot_response,
            self.dialog_eos,
        ]

        attention_mask = [1] * len(input_sequence)

        return {
            "input_ids": input_sequence,
            "attention_mask": attention_mask,
        }


class FoCusDatasetV1:
    def __init__(
        self,
        input_dataset_path: str = None,
    ) -> None:
        assert input_dataset_path is not None, "input_dataset_path is None"

        self.input_dataset_path: str = input_dataset_path
        self.dataset: List[FoCusDatasetSampleDictV1] = []

        self.__build_dataset()

    def __build_dataset(self) -> None:
        initial_dataset = self.__read_dataset(self.input_dataset_path)
        self.dataset = self.__create_initial_dataset(initial_dataset=initial_dataset)

    def __create_initial_dataset(
        self,
        initial_dataset: Dict = None,
    ) -> List[FoCusDatasetSampleDictV1]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for dialog_set in initial_dataset_data:
            persona = dialog_set["persona"]
            utterances = dialog_set["utterance"]
            knowledge = dialog_set["knowledge"]

            for utterance in utterances:
                persona_grounding = list(map(int, utterance["persona_grounding"]))
                knowledge_candidates = utterance["knowledge_candidates"]
                knowledge_answer_index = utterance["knowledge_answer_index"]
                dialog_index_key = [
                    item for item in utterance.keys() if "dialog" in item
                ][0]
                dialog = utterance[dialog_index_key]

                data_sample = FoCusDatasetSampleV1(
                    persona=persona,
                    knowledge_candidates=knowledge_candidates,
                    persona_grounding=persona_grounding,
                    dialog=dialog,
                    knowledge_answer_index=knowledge_answer_index,
                    knowledge=knowledge,
                )
                data_sample = data_sample.get_dict()
                dataset.append(data_sample)

        return dataset

    def __read_dataset(self, input_path: str) -> list:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusDatasetSampleDictV1:
        return self.dataset[index]


class PytorchFoCusDatasetV1(Dataset):
    def __init__(
        self,
        dataset: FoCusDatasetV1,
        tokenizer: BartFoCusTokenizerV1 = None,
        hyperparameters: BartHyperparametersV1 = None,
    ) -> None:
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusDatasetSampleDictV1:
        dataset_sample = self.dataset[index]
        train_sample = BartFoCusDatasetSampleV1(
            focus_dataset_sample=dataset_sample,
            tokenizer=self.tokenizer,
            h_params=self.hyperparameters,
        )
        train_sample = train_sample.get_dict()
        return train_sample


class FoCusLightningDataModuleV1(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str = None,
        valid_path_dataset: str = None,
        hyperparameters: BartHyperparametersV1 = None,
        tokenizer: BartFoCusTokenizerV1 = None,
        is_debug: bool = False,
    ) -> None:
        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.train_dataset = None
        self.valid_dataset = None

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.is_debug = is_debug

    def setup(self, stage: str = None) -> None:
        train_dataset = FoCusDatasetV1(input_dataset_path=self.train_path_dataset)
        valid_dataset = FoCusDatasetV1(input_dataset_path=self.valid_path_dataset)

        if self.is_debug:
            train_dataset = train_dataset[:2]
            valid_dataset = valid_dataset[:2]

        self.train_dataset = PytorchFoCusDatasetV1(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
        )
        self.valid_dataset = PytorchFoCusDatasetV1(
            dataset=valid_dataset,
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hyperparameters.train_batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hyperparameters.valid_batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: List[FoCusDatasetSampleDictV1]) -> Dict:
        max_len = 0
        for item in batch:
            max_len = max(max_len, len(item["input_ids"]))

        pad_input_ids = []
        pad_attention_mask = []

        for item in batch:
            input_ids: List = item["input_ids"]
            attention_mask: List = item["attention_mask"]

            pad_tokens = [self.tokenizer.pad_token_id] * (max_len - len(input_ids))
            pad_attention = [0] * (max_len - len(attention_mask))

            input_ids.extend(pad_tokens)
            attention_mask.extend(pad_attention)

            pad_input_ids.append(input_ids)
            pad_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.tensor(pad_input_ids),
            "attention_mask": torch.tensor(pad_attention_mask),
        }
