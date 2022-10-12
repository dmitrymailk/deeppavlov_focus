import json
from typing import Dict, List, TypedDict


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


class FoCusDatasetKnowledgeSampleDictV1(TypedDict):
    """
    knowledge_candidate: str кандидат на знание
    dialog: List[int] пары диалогов истории
    knowledge_candidate_usage: int 0 или 1 показыват использовалилось ли знание
        в генерации ответа
    knowledge: List[str] все знания об объекте из википедии что у нас есть
    unique_id: str уникальный id сформированный из id диалога и id utterance
    persona: List[str] персона
    """

    knowledge_candidate: str
    dialog: List[str]
    knowledge_candidate_usage: int
    knowledge: List[str]
    unique_id: str
    persona: List[str]
    persona_grounding: List[int]


class FoCusDatasetKnowledgeV1:
    def __init__(
        self,
        input_dataset_path: str,
    ) -> None:
        assert input_dataset_path is not None, "input_dataset_path is None"

        self.input_dataset_path: str = input_dataset_path
        self.dataset: List[FoCusDatasetKnowledgeSampleDictV1] = []

        self.__build_dataset()

    def __build_dataset(self) -> None:
        initial_dataset = self.__read_dataset(self.input_dataset_path)
        self.dataset = self.__create_initial_dataset(initial_dataset=initial_dataset)

    def __create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[FoCusDatasetKnowledgeSampleDictV1]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for dialog_set in initial_dataset_data:
            utterances = dialog_set["utterance"]
            knowledge = dialog_set["knowledge"]
            dialog_id = dialog_set["dialogID"]
            persona = dialog_set["persona"]

            for utterance in utterances:
                knowledge_candidates = utterance["knowledge_candidates"]
                knowledge_answer_index = utterance["knowledge_answer_index"]
                persona_grounding = list(map(int, utterance["persona_grounding"]))
                dialog_index_key = [
                    item for item in utterance.keys() if "dialog" in item
                ][0]
                dialog = utterance[dialog_index_key]
                unique_id = f"{dialog_id}_{dialog_index_key}"

                for i, knowledge_candidate in enumerate(knowledge_candidates):
                    is_used = int(i == knowledge_answer_index)
                    data_sample = FoCusDatasetKnowledgeSampleDictV1(
                        knowledge_candidate=knowledge_candidate,
                        dialog=dialog,
                        knowledge_candidate_usage=is_used,
                        knowledge=knowledge,
                        unique_id=unique_id,
                        persona=persona,
                        persona_grounding=persona_grounding,
                    )

                    dataset.append(data_sample)

        return dataset

    def __read_dataset(self, input_path: str) -> Dict:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusDatasetKnowledgeSampleDictV1:
        return self.dataset[index]


class FoCusDatasetKnowledgeV2:
    def __init__(
        self,
        input_dataset_path: str,
        is_train: bool,
    ) -> None:
        assert input_dataset_path is not None, "input_dataset_path is None"

        self.input_dataset_path: str = input_dataset_path
        self.dataset: List[FoCusDatasetKnowledgeSampleDictV1] = []
        self.is_train = is_train
        self.__build_dataset()

    def __build_dataset(self) -> None:
        initial_dataset = self.__read_dataset(self.input_dataset_path)
        self.dataset = self.__create_initial_dataset(initial_dataset=initial_dataset)

    def __create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[FoCusDatasetKnowledgeSampleDictV1]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for dialog_set in initial_dataset_data:
            utterances = dialog_set["utterance"]
            knowledge = dialog_set["knowledge"]
            dialog_id = dialog_set["dialogID"]
            persona = dialog_set["persona"]

            for utterance in utterances:
                knowledge_candidates = utterance["knowledge_candidates"]
                knowledge_answer_index = utterance["knowledge_answer_index"]
                persona_grounding = list(map(int, utterance["persona_grounding"]))
                dialog_index_key = [
                    item for item in utterance.keys() if "dialog" in item
                ][0]
                dialog = utterance[dialog_index_key]
                unique_id = f"{dialog_id}_{dialog_index_key}"

                has_positive = False
                has_negative = False
                for i, knowledge_candidate in enumerate(knowledge_candidates):
                    is_used = int(i == knowledge_answer_index)
                    data_sample = FoCusDatasetKnowledgeSampleDictV1(
                        knowledge_candidate=knowledge_candidate,
                        dialog=dialog,
                        knowledge_candidate_usage=is_used,
                        knowledge=knowledge,
                        unique_id=unique_id,
                        persona=persona,
                        persona_grounding=persona_grounding,
                    )
                    if self.is_train:
                        if i == knowledge_answer_index:
                            has_positive = True
                            dataset.append(data_sample)

                        if not i == knowledge_answer_index and not has_negative:
                            has_negative = True
                            dataset.append(data_sample)

                        if has_positive and has_negative:
                            break
                    else:
                        dataset.append(data_sample)

        return dataset

    def __read_dataset(self, input_path: str) -> Dict:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusDatasetKnowledgeSampleDictV1:
        return self.dataset[index]


class FoCusDatasetPersonaSampleDictV1(TypedDict):
    """
    persona: List[str] предложения из персоны
    knowledge_candidate: str кандидат который использовался для генерации ответа
    dialog: List[int] пары диалогов истории
    persona_grounding: List[int] 0 или 1 показыват использовалась ли персона
        в генерации ответа
    """

    persona: List[str]
    used_knowledge: str
    dialog: List[str]
    persona_grounding: List[int]


class FoCusDatasetPersonaSampleDictV2(TypedDict):
    """
    persona: str предложение из персоны
    knowledge_candidate: str кандидат который использовался для генерации ответа
    dialog: List[int] пары диалогов истории
    persona_grounding: int 0 или 1 показыват использовалась ли персона
        в генерации ответа
    """

    persona: str
    used_knowledge: str
    dialog: List[str]
    persona_grounding: int
    unique_id: str


class FoCusDatasetPersonaV1:
    def __init__(
        self,
        input_dataset_path: str,
    ) -> None:
        assert input_dataset_path is not None, "input_dataset_path is None"

        self.input_dataset_path: str = input_dataset_path
        self.dataset: List[FoCusDatasetPersonaSampleDictV1] = []
        self.__build_dataset()

    def __build_dataset(self) -> None:
        initial_dataset = self.__read_dataset(self.input_dataset_path)
        self.dataset = self.__create_initial_dataset(initial_dataset=initial_dataset)

    def __create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[FoCusDatasetPersonaSampleDictV1]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for dialog_set in initial_dataset_data:
            utterances = dialog_set["utterance"]
            persona = dialog_set["persona"]

            for utterance in utterances:
                knowledge_candidates = utterance["knowledge_candidates"]
                knowledge_answer_index = utterance["knowledge_answer_index"]
                persona_grounding = list(map(int, utterance["persona_grounding"]))
                dialog_index_key = [
                    item for item in utterance.keys() if "dialog" in item
                ][0]
                dialog = utterance[dialog_index_key]
                true_knowledge = knowledge_candidates[knowledge_answer_index]
                sample = FoCusDatasetPersonaSampleDictV1(
                    persona=persona,
                    used_knowledge=true_knowledge,
                    dialog=dialog,
                    persona_grounding=persona_grounding,
                )
                dataset.append(sample)

        return dataset

    def __read_dataset(self, input_path: str) -> Dict:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusDatasetPersonaSampleDictV1:
        return self.dataset[index]


class FoCusDatasetPersonaV2:
    def __init__(
        self,
        input_dataset_path: str,
        is_train: bool,
    ) -> None:
        assert input_dataset_path is not None, "input_dataset_path is None"

        self.input_dataset_path: str = input_dataset_path
        self.dataset: List[FoCusDatasetPersonaSampleDictV2] = []
        self.is_train = is_train
        self.__build_dataset()

    def __build_dataset(self) -> None:
        initial_dataset = self.__read_dataset(self.input_dataset_path)
        self.dataset = self.__create_initial_dataset(initial_dataset=initial_dataset)

    def __create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[FoCusDatasetPersonaSampleDictV2]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for dialog_set in initial_dataset_data:
            utterances = dialog_set["utterance"]
            dialog_id = dialog_set["dialogID"]
            persona = dialog_set["persona"]

            for utterance in utterances:
                knowledge_candidates = utterance["knowledge_candidates"]
                knowledge_answer_index = utterance["knowledge_answer_index"]
                used_knowledge = knowledge_candidates[knowledge_answer_index]
                persona_grounding = list(map(int, utterance["persona_grounding"]))
                dialog_index_key = [
                    item for item in utterance.keys() if "dialog" in item
                ][0]
                dialog = utterance[dialog_index_key]
                unique_id = f"{dialog_id}_{dialog_index_key}"

                has_positive = False
                has_negative = False
                for i, grounding in enumerate(persona_grounding):
                    if self.is_train:
                        if has_positive and has_negative:
                            break
                        if grounding == 1 and not has_positive:
                            has_positive = True
                            sample = FoCusDatasetPersonaSampleDictV2(
                                persona=persona[i],
                                used_knowledge=used_knowledge,
                                dialog=dialog,
                                persona_grounding=grounding,
                                unique_id=unique_id,
                            )
                            dataset.append(sample)

                        elif grounding == 0 and not has_negative:
                            has_negative = True
                            sample = FoCusDatasetPersonaSampleDictV2(
                                persona=persona[i],
                                used_knowledge=used_knowledge,
                                dialog=dialog,
                                persona_grounding=grounding,
                                unique_id=unique_id,
                            )
                            dataset.append(sample)
                    else:
                        sample = FoCusDatasetPersonaSampleDictV2(
                            persona=persona[i],
                            used_knowledge=used_knowledge,
                            dialog=dialog,
                            persona_grounding=grounding,
                            unique_id=unique_id,
                        )
                        dataset.append(sample)

        return dataset

    def __read_dataset(self, input_path: str) -> Dict:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusDatasetPersonaSampleDictV2:
        return self.dataset[index]


class FoCusTestDatasetSampleDictV1(TypedDict):
    """
    persona: List[str] список предложений из персоны
    knowledge_candidates: List[str] список кандидатов с негативными примерами
        и одним правильным
    query: str последний вопрос от пользователя
    knowledge: List[str] все знания об объекте из википедии что у нас есть
    dialog_id: str идентификатор диалога
    """

    persona: List[str]
    knowledge: List[str]
    knowledge_candidates: List[str]
    query: str
    dialog_id: str
    position: int


class FoCusTestDatasetV1:
    def __init__(
        self,
        input_dataset_path: str,
    ) -> None:
        assert input_dataset_path is not None, "input_dataset_path is None"

        self.input_dataset_path: str = input_dataset_path
        self.dataset: List[FoCusTestDatasetSampleDictV1] = []

        self.__build_dataset()

    def __build_dataset(self) -> None:
        initial_dataset = self.__read_dataset(self.input_dataset_path)
        self.dataset = self.__create_initial_dataset(initial_dataset=initial_dataset)

    def __create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[FoCusTestDatasetSampleDictV1]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for dialog_set in initial_dataset_data:
            utterances = dialog_set["utterance"]
            knowledge = dialog_set["knowledge"]
            dialog_id = dialog_set["dialogID"]

            for i, utterance in enumerate(utterances):
                persona = utterance["persona_candidate"]
                knowledge_candidates = utterance["knowledge_candidate"]
                dialog_key = [item for item in utterance.keys() if "dialog" in item][0]
                query = utterance[dialog_key]

                sample = FoCusTestDatasetSampleDictV1(
                    persona=persona,
                    knowledge=knowledge,
                    knowledge_candidates=knowledge_candidates,
                    query=query,
                    dialog_id=dialog_id,
                    position=i,
                )
                dataset.append(sample)

        return dataset

    def __read_dataset(self, input_path: str) -> Dict:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FoCusTestDatasetSampleDictV1:
        return self.dataset[index]
