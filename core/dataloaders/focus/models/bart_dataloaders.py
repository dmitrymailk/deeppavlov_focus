from itertools import chain
from typing import List, TypedDict

from core.dataloaders.focus.focus_dataloader import FoCusDatasetSampleDictV1
from core.dataloaders.focus.focus_dataloader import FoCusDatasetV1
from core.hyperparameters.bart_hyperparameters import BartHyperparametersV3
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV2
from core.utils import FoCusTfIdf

from torch.utils.data import Dataset


class BartFoCusDatasetSampleDictV2(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    persona_grounding: List[int]
    knowledge_answer_index: int
    persona_sep_index: int
    knowledge_sep_index: int
    dialog_bos_index: int
    dialog_eos_index: int


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


class BartFoCusDatasetSampleDictV4(TypedDict):
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


class BartFoCusDatasetSampleDictV5(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]

    labels: List[int]
    persona_grounding: List[int]
    knowledge_candidates_answer_index: int

    persona_ids: List[int]
    persona_attention_mask: List[int]

    knowledge_ids: List[int]
    knowledge_attention_mask: List[int]

    knowledge_candidates_ids: List[int]
    knowledge_candidates_attention_mask: List[int]

    query_ids: List[int]
    query_attention_mask: List[int]


class BartFoCusDatasetSampleV4:
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

    def __flat_list(self, list_of_lists: List[List] | List) -> List:
        return list(chain.from_iterable(list_of_lists))

    def get_dict(self) -> BartFoCusDatasetSampleDictV3:
        """
        Returns:
            input_ids (List[int]):
                [BOS][persona][SEP][knowledge_candidate][SEP]<query>[dialog][-2]</query>[EOS]
            labels (List[int]):
                [BOS]<response>[dialog][-1]</response>[EOS]

                [persona] - это предложения которые точно использовались для ответа
                query - это последний вопрос от пользователя
                response - это ответ от бота за запрос пользователя
                [knowledge_candidate] - это предложение из базы которое точно
                    использовалось

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

        max_persona_tokens = self.h_params.max_persona_tokens
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_candidates_tokens = self.h_params.max_knowledge_candidates_tokens

        persona = self.focus_dataset_sample["persona"]
        dialog = self.focus_dataset_sample["dialog"]
        # knowledge = self.focus_dataset_sample["knowledge"]
        persona_grounding = self.focus_dataset_sample["persona_grounding"]
        knowledge_candidates_answer_index = self.focus_dataset_sample[
            "knowledge_answer_index"
        ]
        knowledge_candidates = self.focus_dataset_sample["knowledge_candidates"]

        # persona
        persona = [
            persona_item
            for persona_item, persona_grounding_item in zip(persona, persona_grounding)
            if persona_grounding_item == 1
        ]
        if len(persona) > 0:
            persona = self.tokenizer.batch_encode_plus(
                persona,
                add_special_tokens=False,
                truncation=True,
                max_length=max_persona_tokens,
            )
            persona = persona["input_ids"]

        # knowledge_candidates
        used_knowledge_candidates = [
            knowledge_candidates[knowledge_candidates_answer_index],
        ]
        used_knowledge_candidates = self.tokenizer.batch_encode_plus(
            used_knowledge_candidates,
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_candidates_tokens,
        )

        # query
        query = dialog[-2]
        query = self.tokenizer.batch_encode_plus(
            [query],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        # response
        response = dialog[-1]
        response = self.tokenizer.batch_encode_plus(
            [response],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )

        flat_persona = self.__flat_list(persona)  # type: ignore
        flat_knowledge_candidates = self.__flat_list(
            used_knowledge_candidates["input_ids"],  # type: ignore
        )
        flat_query = self.__flat_list(query["input_ids"])  # type: ignore
        flat_responce = self.__flat_list(response["input_ids"])  # type: ignore
        # [BOS][persona][SEP][knowledge_candidate][SEP]<query>[dialog][-2]</query>[EOS]
        input_ids = [
            self.bos_token_id,
            *flat_persona,
            self.sep_token_id,
            *flat_knowledge_candidates,
            self.sep_token_id,
            self.query_bos_id,
            *flat_query,
            self.query_eos_id,
            self.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)
        # [BOS]<response>[dialog][-1]</response>[EOS]
        labels = [
            self.bos_token_id,
            self.response_bos_id,
            *flat_responce,
            self.response_eos_id,
            self.eos_token_id,
        ]
        persona_sep_index = input_ids.index(self.sep_token_id)
        knowledge_candidates_sep_index = (
            len(input_ids) - 1 - input_ids[::-1].index(self.sep_token_id)
        )
        query_bos_index = input_ids.index(self.query_bos_id)
        query_eos_index = input_ids.index(self.query_eos_id)
        bos_index = 0
        eos_index = len(input_ids) - 1

        return BartFoCusDatasetSampleDictV4(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            persona_grounding=persona_grounding,
            knowledge_candidates_answer_index=knowledge_candidates_answer_index,
            persona_sep_index=persona_sep_index,
            knowledge_candidates_sep_index=knowledge_candidates_sep_index,
            query_eos_index=query_eos_index,
            query_bos_index=query_bos_index,
            bos_index=bos_index,
            eos_index=eos_index,
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


class PytorchFoCusDatasetV4(Dataset):
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

    def __getitem__(self, index: int) -> BartFoCusDatasetSampleDictV4:
        dataset_sample = self.dataset[index]
        train_sample = BartFoCusDatasetSampleV4(
            focus_dataset_sample=dataset_sample,
            tokenizer=self.tokenizer,
            h_params=self.hyperparameters,
        )
        train_sample = train_sample.get_dict()
        return train_sample


class BartFoCusDatasetSampleV5:
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

    def __flat_list(self, list_of_lists: List[List] | List) -> List:
        return list(chain.from_iterable(list_of_lists))

    def get_dict(self) -> BartFoCusDatasetSampleDictV5:
        """
        Returns:
            input_ids (List[int]):
                [BOS][persona][knowledge]<query>[dialog][-2]</query>[EOS]
                persona - это сконкатенированная персона
                knowledge - наиболее похожее предложение из базы знаний
                    на query
                query - это вопрос, который задал пользователь

            labels (List[int]):
                [BOS]<response>[dialog][-1]</response>[EOS]

            knowledge_candidate_ids (List[int]):
                [BOS][knowledge_candidate][EOS]

            knowledge_ids (List[int]):
                [BOS][knowledge][EOS]

            persona_ids (List[int]):
                [BOS][persona][EOS]

            классификацию knowledge_candidates на основе:
                - [EOS] из knowledge_candidates
                - [EOS] из knowledge
                - [EOS] из persona
                - [EOS] из query

            классификацию persona на основе:
                - [EOS] из knowledge
                - [EOS] из persona
                - [EOS] из query
        """

        dialog_history_length = self.h_params.dialog_history_length
        assert dialog_history_length == 1

        max_persona_tokens = self.h_params.max_persona_tokens
        max_dialog_history_tokens = self.h_params.max_dialog_history_tokens
        max_knowledge_tokens = self.h_params.max_knowledge_tokens

        persona = self.focus_dataset_sample["persona"]
        dialog = self.focus_dataset_sample["dialog"]
        knowledge = self.focus_dataset_sample["knowledge"]
        persona_grounding = self.focus_dataset_sample["persona_grounding"]
        knowledge_candidates_answer_index = self.focus_dataset_sample[
            "knowledge_answer_index"
        ]
        knowledge_candidates = self.focus_dataset_sample["knowledge_candidates"]

        # persona
        persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
            max_length=max_persona_tokens,
        )
        persona_ids = persona["input_ids"]
        persona_ids = [[*item, self.sep_token_id] for item in persona_ids]  # type: ignore

        # knowledge_candidates
        knowledge_candidates = self.tokenizer.batch_encode_plus(
            knowledge_candidates,
            add_special_tokens=False,
            truncation=True,
            max_length=100,
        )
        knowledge_candidates_ids = knowledge_candidates["input_ids"]
        knowledge_candidates_ids = [
            [*item, self.sep_token_id] for item in knowledge_candidates_ids  # type: ignore
        ]

        # query
        query = dialog[-2]
        query = self.tokenizer.batch_encode_plus(
            [query],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        query_ids = query["input_ids"]
        # response
        response = dialog[-1]
        response = self.tokenizer.batch_encode_plus(
            [response],
            add_special_tokens=False,
            truncation=True,
            max_length=max_dialog_history_tokens,
        )
        response_ids = response["input_ids"]

        # knowledge
        knowledge = self.tokenizer.batch_encode_plus(
            knowledge,
            add_special_tokens=False,
            truncation=True,
            max_length=max_knowledge_tokens,
        )
        knowledge_ids = knowledge["input_ids"]
        tf_idf = FoCusTfIdf(corpus=knowledge_ids)
        most_similar_knowledge_candidates = tf_idf.top_similar(
            query=query_ids,  # type: ignore
        )
        most_similar_knowledge_candidates = [
            [*item, self.sep_token_id] for item in most_similar_knowledge_candidates
        ]

        flat_persona = self.__flat_list(persona_ids)  # type: ignore
        flat_query = self.__flat_list(query_ids)  # type: ignore
        flat_responce = self.__flat_list(response_ids)  # type: ignore
        flat_knowledge = self.__flat_list(most_similar_knowledge_candidates)
        flat_knowledge_candidates = self.__flat_list(knowledge_candidates_ids)

        # [BOS][persona][knowledge]<query>[dialog][-2]</query>[EOS]
        input_ids = [
            self.bos_token_id,
            *flat_persona,
            *flat_knowledge,
            self.query_bos_id,
            *flat_query,
            self.query_eos_id,
            self.eos_token_id,
        ]

        # [BOS]<response>[dialog][-1]</response>[EOS]
        labels = [
            self.bos_token_id,
            self.response_bos_id,
            *flat_responce,
            self.response_eos_id,
            self.eos_token_id,
        ]

        flat_knowledge = [
            self.bos_token_id,
            *flat_knowledge,
            self.eos_token_id,
        ]

        flat_persona = [
            self.bos_token_id,
            *flat_persona,
            self.eos_token_id,
        ]

        flat_query = [
            self.bos_token_id,
            *flat_query,
            self.eos_token_id,
        ]

        flat_knowledge_candidates = [
            self.bos_token_id,
            *flat_knowledge_candidates,
            self.eos_token_id,
        ]

        attention_mask = [1] * len(input_ids)
        knowledge_attention_mask = [1] * len(flat_knowledge)
        persona_attention_mask = [1] * len(flat_persona)
        query_attention_mask = [1] * len(flat_query)
        knowledge_candidates_attention_mask = [1] * len(flat_knowledge_candidates)

        return BartFoCusDatasetSampleDictV5(
            input_ids=input_ids,
            labels=labels,
            knowledge_ids=flat_knowledge,
            persona_ids=flat_persona,
            query_ids=flat_query,
            knowledge_candidates_ids=flat_knowledge_candidates,
            attention_mask=attention_mask,
            knowledge_attention_mask=knowledge_attention_mask,
            persona_attention_mask=persona_attention_mask,
            query_attention_mask=query_attention_mask,
            knowledge_candidates_answer_index=knowledge_candidates_answer_index,
            knowledge_candidates_attention_mask=knowledge_candidates_attention_mask,
            persona_grounding=persona_grounding,
        )
