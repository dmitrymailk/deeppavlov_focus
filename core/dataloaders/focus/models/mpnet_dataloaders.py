from typing import List, TypedDict

from core.dataloaders.focus.focus_dataloader import (
    FoCusDatasetKnowledgeSampleDictV3,
    FoCusDatasetPersonaSampleDictV2,
    FoCusDatasetKnowledgeSampleDictV2,
)
from core.hyperparameters.mpnet_hyperparameters import MPNetHyperparametersV1
from core.utils import flat_list

from transformers import AutoTokenizer  # type: ignore


class MPNetV3FoCusPersonaDatasetSampleDictV1(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    labels: int


class MPNetFoCusPersonaDatasetSampleV1:
    def __init__(
        self,
        dataset_sample: FoCusDatasetPersonaSampleDictV2,
        tokenizer: AutoTokenizer,
        h_params: MPNetHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: AutoTokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> MPNetV3FoCusPersonaDatasetSampleDictV1:
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

        return MPNetV3FoCusPersonaDatasetSampleDictV1(
            input_ids=input_ids,
            labels=persona_grounding,
            attention_mask=attention_mask,
        )


class MPNetFoCusPersonaTestDatasetSampleDictV1(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]


class MPNetFoCusPersonaTestDatasetSampleDictV2(TypedDict):
    persona_sentence: str
    used_knowledge: str
    query: str


class MPNetFoCusPersonaTestDatasetSampleV1:
    def __init__(
        self,
        dataset_sample: MPNetFoCusPersonaTestDatasetSampleDictV2,
        tokenizer: AutoTokenizer,
        h_params: MPNetHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: AutoTokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> MPNetFoCusPersonaTestDatasetSampleDictV1:

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

        encoded_query = self.tokenizer.batch_encode_plus(  # type: ignore
            [query],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        encoded_query = flat_list(encoded_query["input_ids"])

        input_ids = [
            self.bos_token_id,
            *encoded_persona,
            *encoded_knowledge,
            *encoded_query,
            self.eos_token_id,
        ]
        attention_mask = len(input_ids) * [1]

        return MPNetFoCusPersonaTestDatasetSampleDictV1(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class MPNetV3FoCusKnowledgeDatasetSampleDictV1(TypedDict):
    sentence_1: List[int]
    sentence_2: List[int]
    attention_mask_1: List[int]
    attention_mask_2: List[int]
    utterance_id: str
    score: float


class MPNetFoCusKnowledgeDatasetSampleV1:
    def __init__(
        self,
        dataset_sample: FoCusDatasetKnowledgeSampleDictV2,
        tokenizer: AutoTokenizer,
        h_params: MPNetHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: AutoTokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> MPNetV3FoCusKnowledgeDatasetSampleDictV1:

        max_length = self.tokenizer.model_max_length  # type: ignore

        knowledge_candidate = self.dataset_sample["knowledge_candidate"]
        query = self.dataset_sample["query"]
        utterance_id = self.dataset_sample["utterance_id"]
        score = self.dataset_sample["score"]

        encoded_knowledge = self.tokenizer.batch_encode_plus(  # type: ignore
            [knowledge_candidate],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        encoded_knowledge = flat_list(encoded_knowledge["input_ids"])

        encoded_query = self.tokenizer.batch_encode_plus(  # type: ignore
            [query],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        encoded_query = flat_list(encoded_query["input_ids"])

        sentence_1 = encoded_knowledge
        sentence_2 = encoded_query
        attention_mask_1 = len(sentence_1) * [1]
        attention_mask_2 = len(sentence_2) * [1]

        return MPNetV3FoCusKnowledgeDatasetSampleDictV1(
            sentence_1=sentence_1,
            sentence_2=sentence_2,
            attention_mask_1=attention_mask_1,
            attention_mask_2=attention_mask_2,
            utterance_id=utterance_id,
            score=score,
        )


class MPNetV3FoCusKnowledgeDatasetSampleDictV2(TypedDict):
    source: List[int]
    positive: List[int]
    negative: List[int]
    source_mask: List[int]
    positive_mask: List[int]
    negative_mask: List[int]
    utterance_id: str


class MPNetFoCusKnowledgeDatasetSampleV2:
    def __init__(
        self,
        dataset_sample: FoCusDatasetKnowledgeSampleDictV3,
        tokenizer: AutoTokenizer,
        h_params: MPNetHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer: AutoTokenizer = tokenizer
        self.h_params = h_params

        self.bos_token_id = self.tokenizer.bos_token_id  # type: ignore
        self.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

    def get_dict(self) -> MPNetV3FoCusKnowledgeDatasetSampleDictV2:

        max_length = self.tokenizer.model_max_length  # type: ignore

        source = self.dataset_sample["source"]
        positive = self.dataset_sample["positive"]
        negative = self.dataset_sample["negative"]
        utterance_id = self.dataset_sample["utterance_id"]

        source = self.tokenizer.batch_encode_plus(  # type: ignore
            [source],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        source = flat_list(source["input_ids"])

        positive = self.tokenizer.batch_encode_plus(  # type: ignore
            [positive],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        positive = flat_list(positive["input_ids"])

        negative = self.tokenizer.batch_encode_plus(  # type: ignore
            [negative],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        negative = flat_list(negative["input_ids"])

        source_mask = len(source) * [1]
        positive_mask = len(positive) * [1]
        negative_mask = len(negative) * [1]

        return MPNetV3FoCusKnowledgeDatasetSampleDictV2(
            source=source,
            positive=positive,
            negative=negative,
            source_mask=source_mask,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
            utterance_id=utterance_id,
        )
