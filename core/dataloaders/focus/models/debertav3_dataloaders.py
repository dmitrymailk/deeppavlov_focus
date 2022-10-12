from typing import List, TypedDict

from core.dataloaders.focus.focus_dataloader import (
    FoCusDatasetKnowledgeSampleDictV1,
    FoCusDatasetKnowledgeV1,
    FoCusDatasetKnowledgeV2,
    FoCusDatasetPersonaSampleDictV1,
    FoCusDatasetPersonaSampleDictV2,
)
from core.hyperparameters.debertav3_hyperparameters import DebertaV3HyperparametersV1
from core.utils import flat_list

from torch.utils.data import Dataset


from transformers.utils.dummy_sentencepiece_objects import DebertaV2Tokenizer


class DebertaV3FoCusKnowledgeDatasetSampleDictV1(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    labels: int
    unique_id: str


class DebertaV3FoCusPersonaDatasetSampleDictV1(TypedDict):
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[int]
    max_len: int


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


class DebertaV3FoCusKnowledgeDatasetSampleV2:
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
                [BOS][used_persona][knowledge_candidate][dialog][-2][EOS]
                [dialog][-2] - это последний вопрос от пользователя
                [knowledge_candidate] - это кандидат на знание. большая часть
                    будет неправильными ответами
                [used_persona] - это персона, которая была использована для ответа
            labels (int): 0 или 1. ложь или правда. использовалось ли знание для ответа
                или нет.
        """
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens
        max_persona_tokens = self.h_params.max_persona_tokens

        knowledge_candidate = self.dataset_sample["knowledge_candidate"]
        dialog = self.dataset_sample["dialog"]
        knowledge_candidate_usage = self.dataset_sample["knowledge_candidate_usage"]
        unique_id = self.dataset_sample["unique_id"]
        persona = self.dataset_sample["persona"]
        persona_grounding = self.dataset_sample["persona_grounding"]

        used_persona = [
            sent for sent, used in zip(persona, persona_grounding) if used == 1
        ]

        if len(used_persona) > 0:
            used_persona = self.tokenizer.batch_encode_plus(  # type: ignore
                used_persona,
                add_special_tokens=False,
                truncation=True,
                max_length=max_persona_tokens,
            )
            used_persona = used_persona["input_ids"]
            used_persona = flat_list(used_persona)

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
            *used_persona,
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


class DebertaV3FoCusKnowledgeDatasetSampleV3:
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
                [BOS][knowledge_candidate][dialog][EOS]
                [dialog] - это последний вопрос от пользователя, предыдущий
                    ответ бота и предыдущий вопрос пользователя
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

        encoded_dialog = self.tokenizer.batch_encode_plus(  # type: ignore
            dialog[-4:-1],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        encoded_dialog = encoded_dialog["input_ids"]
        encoded_dialog = flat_list(encoded_dialog)

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


class DebertaV3FoCusKnowledgeDatasetSampleV4:
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
                [BOS][persona][knowledge_candidate][dialog][EOS]
                [dialog] - это последний вопрос от пользователя, предыдущий
                    ответ бота и предыдущий вопрос пользователя
                [knowledge_candidate] - это кандидат на знание. большая часть
                    будет неправильными ответами
                [persona] - это все 5 предложений персоны
            labels (int): 0 или 1. ложь или правда. использовалось ли знание для ответа
                или нет.
        """
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens
        max_persona_tokens = self.h_params.max_persona_tokens

        knowledge_candidate = self.dataset_sample["knowledge_candidate"]
        dialog = self.dataset_sample["dialog"]
        knowledge_candidate_usage = self.dataset_sample["knowledge_candidate_usage"]
        unique_id = self.dataset_sample["unique_id"]
        persona = self.dataset_sample["persona"]

        persona = " ".join(persona)

        encoded_persona = self.tokenizer.batch_encode_plus(  # type: ignore
            [persona],
            add_special_tokens=False,
            truncation=True,
            max_length=max_persona_tokens,
        )

        encoded_knowledge_candidate = self.tokenizer.encode(  # type: ignore
            knowledge_candidate,
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_candidates_tokens,
        )

        encoded_dialog = self.tokenizer.batch_encode_plus(  # type: ignore
            dialog[-4:-1],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        encoded_dialog = encoded_dialog["input_ids"]
        encoded_dialog = flat_list(encoded_dialog)

        encoded_persona = encoded_persona["input_ids"]
        encoded_persona = flat_list(encoded_persona)

        input_ids = [
            self.bos_token_id,
            *encoded_persona,
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


class DebertaV3FoCusPersonaDatasetSampleV1:
    def __init__(
        self,
        dataset_sample: FoCusDatasetPersonaSampleDictV1,
        tokenizer: DebertaV2Tokenizer,
        h_params: DebertaV3HyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: DebertaV2Tokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> DebertaV3FoCusPersonaDatasetSampleDictV1:
        """
        Returns:
            input_ids (List[List[int]]):
                List[int]
                    [BOS][persona_sentence][used_knowledge][query][EOS]
                    [query] - это последний вопрос от пользователя
                    [used_knowledge] - это знание которое точно использовалось для
                        генерации ответа
                    [persona_sentence] - это одно из 5 предложений персоны
            labels (List[int]): 0 или 1. ложь или правда. использовалась ли персона
                для ответа или нет.
        """
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens
        max_persona_tokens = self.h_params.max_persona_tokens

        # persona: List[str]
        # used_knowledge: str
        # dialog: List[str]
        # persona_grounding: List[int]

        used_knowledge = self.dataset_sample["used_knowledge"]
        dialog = self.dataset_sample["dialog"]
        persona = self.dataset_sample["persona"]
        persona_grounding = self.dataset_sample["persona_grounding"]

        encoded_persona = self.tokenizer.batch_encode_plus(  # type: ignore
            persona,
            add_special_tokens=False,
            truncation=True,
            max_length=max_persona_tokens,
        )

        encoded_knowledge = self.tokenizer.batch_encode_plus(  # type: ignore
            [used_knowledge],
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_candidates_tokens,
        )
        encoded_knowledge = flat_list(encoded_knowledge["input_ids"])

        encoded_dialog = self.tokenizer.batch_encode_plus(  # type: ignore
            [dialog[-2]],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        encoded_dialog = flat_list(encoded_dialog["input_ids"])

        concatenated_persona = []
        attention_mask = []
        max_len = 0
        for persona_sent in encoded_persona["input_ids"]:
            new_persona_sent = [
                self.bos_token_id,
                *persona_sent,
                *encoded_knowledge,
                *encoded_dialog,
                self.eos_token_id,
            ]
            max_len = max(max_len, len(new_persona_sent))
            mask = len(new_persona_sent) * [1]
            concatenated_persona.append(new_persona_sent)
            attention_mask.append(mask)

        return DebertaV3FoCusPersonaDatasetSampleDictV1(
            input_ids=concatenated_persona,
            labels=persona_grounding,
            attention_mask=attention_mask,
            max_len=max_len,
        )


class DebertaV3FoCusPersonaDatasetSampleDictV2(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    labels: int
    unique_id: str


class DebertaV3FoCusPersonaDatasetSampleV2:
    def __init__(
        self,
        dataset_sample: FoCusDatasetPersonaSampleDictV2,
        tokenizer: DebertaV2Tokenizer,
        h_params: DebertaV3HyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: DebertaV2Tokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> DebertaV3FoCusPersonaDatasetSampleDictV2:
        """
        Returns:
            input_ids (List[int]):
                [BOS][persona_sentence][used_knowledge][query][EOS]
                [query] - это последний вопрос от пользователя
                [used_knowledge] - это знание которое точно использовалось для
                    генерации ответа
                [persona_sentence] - это одно из 5 предложений персоны
            labels (int): 0 или 1. ложь или правда. использовалась ли персона
                для ответа или нет.
        """
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens
        max_persona_tokens = self.h_params.max_persona_tokens

        # persona: str
        # used_knowledge: str
        # dialog: List[str]
        # persona_grounding: int
        # unique_id: str

        used_knowledge = self.dataset_sample["used_knowledge"]
        dialog = self.dataset_sample["dialog"]
        persona = self.dataset_sample["persona"]
        persona_grounding = self.dataset_sample["persona_grounding"]

        encoded_persona = self.tokenizer.batch_encode_plus(  # type: ignore
            [persona],
            add_special_tokens=False,
            truncation=True,
            max_length=max_persona_tokens,
        )
        encoded_persona = flat_list(encoded_persona["input_ids"])

        encoded_knowledge = self.tokenizer.batch_encode_plus(  # type: ignore
            [used_knowledge],
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_candidates_tokens,
        )
        encoded_knowledge = flat_list(encoded_knowledge["input_ids"])

        encoded_dialog = self.tokenizer.batch_encode_plus(  # type: ignore
            [dialog[-2]],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        encoded_dialog = flat_list(encoded_dialog["input_ids"])

        input_ids = [
            self.bos_token_id,
            *encoded_persona,
            *encoded_knowledge,
            *encoded_dialog,
            self.eos_token_id,
        ]
        attention_mask = len(input_ids) * [1]

        return DebertaV3FoCusPersonaDatasetSampleDictV2(
            input_ids=input_ids,
            labels=persona_grounding,
            attention_mask=attention_mask,
            unique_id=self.dataset_sample["unique_id"],
        )


class DebertaV3FoCusPersonaTestDatasetSampleDictV1(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]


class DebertaV3FoCusPersonaTestDatasetSampleDictV2(TypedDict):
    persona_sentence: str
    used_knowledge: str
    query: str


class DebertaV3FoCusPersonaTestDatasetSampleV1:
    def __init__(
        self,
        dataset_sample: DebertaV3FoCusPersonaTestDatasetSampleDictV2,
        tokenizer: DebertaV2Tokenizer,
        h_params: DebertaV3HyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: DebertaV2Tokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> DebertaV3FoCusPersonaTestDatasetSampleDictV1:
        """
        Returns:
            input_ids (List[int]):
                [BOS][persona_sentence][used_knowledge][query][EOS]
                [query] - это последний вопрос от пользователя
                [used_knowledge] - это знание которое точно использовалось для
                    генерации ответа
                [persona_sentence] - это одно из 5 предложений персоны
        """
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens
        max_persona_tokens = self.h_params.max_persona_tokens

        used_knowledge = self.dataset_sample["used_knowledge"]
        query = self.dataset_sample["query"]
        persona_sentence = self.dataset_sample["persona_sentence"]

        encoded_persona = self.tokenizer.batch_encode_plus(  # type: ignore
            [persona_sentence],
            add_special_tokens=False,
            truncation=True,
            max_length=max_persona_tokens,
        )
        encoded_persona = flat_list(encoded_persona["input_ids"])

        encoded_knowledge = self.tokenizer.batch_encode_plus(  # type: ignore
            [used_knowledge],
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_candidates_tokens,
        )
        encoded_knowledge = flat_list(encoded_knowledge["input_ids"])

        encoded_dialog = self.tokenizer.batch_encode_plus(  # type: ignore
            [query],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        encoded_dialog = flat_list(encoded_dialog["input_ids"])

        input_ids = [
            self.bos_token_id,
            *encoded_persona,
            *encoded_knowledge,
            *encoded_dialog,
            self.eos_token_id,
        ]
        attention_mask = len(input_ids) * [1]

        return DebertaV3FoCusPersonaTestDatasetSampleDictV1(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
