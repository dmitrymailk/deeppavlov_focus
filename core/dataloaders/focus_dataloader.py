import json
import os
from itertools import chain
from typing import Dict, List, Optional, TypedDict, cast

from core.hyperparameters.bart_hyperparameters import (
    # BartHyperparametersV1,
    BartHyperparametersV3,
)
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV2
from core.utils import FoCusTfIdf

from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader, Dataset


class FoCusDatasetSampleDictV1(TypedDict):
    """
    persona: List[str] список предложений из персоны
    knowledge_candidates: List[str] список кандидатов с негативными примерами
        и одним правильным
    persona_grounding: List[int] маска которая указывает какие предложения из
        персоны использовались
    dialog: List[int] пары диалогов истории
    knowledge_answer_index: int индекс правильного ответа из кандидатов
    knowledge: List[str] все знания об объекте из википедии что у нас есть
    """

    persona: List[str]
    knowledge_candidates: List[str]
    persona_grounding: List[int]
    dialog: List[str]
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
        return FoCusDatasetSampleDictV1(
            persona=self.persona,
            knowledge_candidates=self.knowledge_candidates,
            persona_grounding=self.persona_grounding,
            dialog=self.dialog,
            knowledge_answer_index=self.knowledge_answer_index,
            knowledge=self.knowledge,
        )


class FoCusDatasetV1:
    def __init__(
        self,
        input_dataset_path: str,
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
        initial_dataset: Dict,
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

    def __read_dataset(self, input_path: str) -> Dict:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusDatasetSampleDictV1:
        return self.dataset[index]


class BartFoCusDatasetSampleDictV2(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    persona_grounding: List[int]
    knowledge_answer_index: int
    persona_sep_index: int
    knowledge_sep_index: int
    dialog_bos_index: int
    dialog_eos_index: int


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


class BartFoCusDatasetSampleDictV3(TypedDict):
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]
    persona_grounding: List[int]
    knowledge_candidates_answer_index: int
    persona_sep_index: int
    knowledge_candidates_sep_index: int
    query_eos_index: int
    query_bos_index: int
    bos_index: int
    eos_index: int


class BartFoCusDatasetSampleV3:
    def __init__(
        self,
        focus_dataset_sample: FoCusDatasetSampleDictV1,
        tokenizer: BartFoCusTokenizerV2,
        h_params: BartHyperparametersV3,
    ) -> None:
        self.focus_dataset_sample = focus_dataset_sample
        self.tokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.response_bos_id = self.__get_token_id(h_params.response_bos_token)
        self.response_eos_id = self.__get_token_id(h_params.response_eos_token)
        self.query_bos_id = self.__get_token_id(h_params.query_bos_token)
        self.query_eos_id = self.__get_token_id(h_params.query_eos_token)
        self.sep_token_id = self.__get_token_id(h_params.sep_token)

    def __get_token_id(self, token: str) -> int:
        token_index = self.tokenizer.get_vocab().get(token, self.unk_token_id)
        return token_index

    def __flat_list(self, list_of_lists: List[List]) -> List:
        return list(chain.from_iterable(list_of_lists))

    def get_dict(self) -> BartFoCusDatasetSampleDictV3:
        """
        Returns:
            input_ids (List[int]):
                [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query><response>[dialog][-1]</response>[EOS]
                [persona] - склееенные предложения персоны, 5шт
                query - это последний вопрос от пользователя
                response - это ответ от бота за запрос пользователя
                [knowledge_candidates] - это топ 2 похожих предложений из
                    knowledge_candidates на query

                классификацию knowledge_candidates на основе:
                    - <query>
                    - </query>
                    - [EOS]
                    - [SEP] после [knowledge_candidates]
                    - [BOS]

                классификацию persona на основе:
                    - <query>
                    - </query>
                    - [EOS]
                    - [SEP] после [persona]
                    - [BOS]
        """

        dialog_history_length = self.h_params.dialog_history_length
        assert dialog_history_length == 1

        context_length = self.h_params.context_length
        max_persona_tokens = self.h_params.max_persona_tokens
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens

        max_full_persona_tokens = self.h_params.max_full_persona_tokens
        max_full_knowledge_candidates_tokens = (
            self.h_params.max_full_knowledge_candidates_tokens
        )

        persona = self.focus_dataset_sample["persona"]
        dialog = self.focus_dataset_sample["dialog"]
        # knowledge = self.focus_dataset_sample["knowledge"]
        persona_grounding = self.focus_dataset_sample["persona_grounding"]
        knowledge_candidates_answer_index = self.focus_dataset_sample[
            "knowledge_answer_index"
        ]
        knowledge_candidates = self.focus_dataset_sample["knowledge_candidates"]

        # persona
        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
            max_length=max_persona_tokens,
        )

        dialog_history = dialog[-2 * dialog_history_length :]
        dialog_query = self.tokenizer.batch_encode_plus(
            dialog_history[:-1],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        dialog_response = self.tokenizer.batch_encode_plus(
            dialog_history[-1:],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )

        # контекст на основе которого подбирается knowledge_candidates
        query_context = dialog_query["input_ids"][-context_length:]  # type: ignore
        encoded_knowledge_candidates = self.tokenizer.batch_encode_plus(
            knowledge_candidates,
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_candidates_tokens,
        )

        tf_idf = FoCusTfIdf(corpus=encoded_knowledge_candidates["input_ids"])
        most_similar_knowledge_candidates = tf_idf.top_similar(
            query=query_context,
        )

        flat_persona = self.__flat_list(encoded_persona["input_ids"])  # type: ignore
        flat_knowledge_candidates = self.__flat_list(most_similar_knowledge_candidates)
        flat_dialog_query = self.__flat_list(dialog_query["input_ids"])  # type: ignore
        flat_dialog_response = self.__flat_list(
            dialog_response["input_ids"],  # type: ignore
        )

        flat_persona = flat_persona[:max_full_persona_tokens]
        flat_knowledge_candidates = flat_knowledge_candidates[
            :max_full_knowledge_candidates_tokens
        ]
        flat_dialog_query = flat_dialog_query[:max_dialog_history_tokens]
        flat_dialog_response = flat_dialog_response[:max_dialog_history_tokens]

        # [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query>[EOS]
        input_ids = [
            self.bos_token_id,
            *flat_persona,
            self.sep_token_id,
            *flat_knowledge_candidates,
            self.sep_token_id,
            self.query_bos_id,
            *flat_dialog_query,
            self.query_eos_id,
            self.eos_token_id,
        ]
        # [BOS]<response>[dialog][-1]</response>[EOS]
        labels = [
            self.bos_token_id,
            self.response_bos_id,
            *flat_dialog_response,
            self.response_eos_id,
            self.eos_token_id,
        ]

        attention_mask = [1] * len(input_ids)
        bos_index = 0
        persona_sep_index = len(flat_persona) + 1
        knowledge_candidates_sep_index = (
            persona_sep_index
            + len(
                flat_knowledge_candidates,
            )
            + 1
        )
        query_bos_index = knowledge_candidates_sep_index + 1
        query_eos_index = query_bos_index + len(flat_dialog_query) + 1
        eos_index = len(input_ids) - 1

        return BartFoCusDatasetSampleDictV3(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            persona_grounding=persona_grounding,
            knowledge_candidates_sep_index=knowledge_candidates_sep_index,
            bos_index=bos_index,
            persona_sep_index=persona_sep_index,
            query_bos_index=query_bos_index,
            query_eos_index=query_eos_index,
            eos_index=eos_index,
            knowledge_candidates_answer_index=knowledge_candidates_answer_index,
        )


class PytorchFoCusDatasetV3(Dataset):
    def __init__(
        self,
        dataset: FoCusDatasetV1,
        tokenizer: BartFoCusTokenizerV2,
        hyperparameters: BartHyperparametersV3,
    ) -> None:
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> BartFoCusDatasetSampleDictV3:
        dataset_sample = self.dataset[index]
        train_sample = BartFoCusDatasetSampleV3(
            focus_dataset_sample=dataset_sample,
            tokenizer=self.tokenizer,
            h_params=self.hyperparameters,
        )
        train_sample = train_sample.get_dict()
        return train_sample


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
