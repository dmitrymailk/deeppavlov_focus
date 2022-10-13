import json
from typing import List, TypedDict

from core.base_models.bart_models import BartLMV7
from core.base_models.debertav3_models import DebertaV3PersonaClassificationV3
from core.dataloaders.focus.focus_dataloader import FoCusTestDatasetV1
from core.dataloaders.focus.models.bart_dataloaders import BartFoCusTestDatasetSampleV1
from core.dataloaders.focus.models.bart_dataloaders import (
    BartRensponseTestDatasetDictV1,
)
from core.dataloaders.focus.models.debertav3_dataloaders import (
    DebertaV3FoCusPersonaTestDatasetSampleDictV2,
    DebertaV3FoCusPersonaTestDatasetSampleV1,
)
from core.hyperparameters.bart_hyperparameters import BartHyperparametersV3
from core.hyperparameters.debertav3_hyperparameters import DebertaV3HyperparametersV1
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV2

from sentence_transformers import SentenceTransformer, util


import torch

from transformers import AutoTokenizer  # type: ignore
from transformers import BartConfig  # type: ignore
from transformers import DebertaV2Config  # type: ignore


test_dataset = FoCusTestDatasetV1(
    input_dataset_path="./datasets/FoCus/test_focus_public.json",
)


class FocusKnowledgeKandidateExtractorDictV1(TypedDict):
    predicted_index: int
    predicted_knowledge: str


class BartFocusTestDatasetDictV1(TypedDict):
    """
    knowledge: List[str] все знания об объекте из википедии что у нас есть
    knowledge_candidates: List[str] 1 истиный пример, который использовался и 9 ложных
    query: str последний вопрос от пользователя
    dialog_id: str идентификатор диалога
    predicted_persona_grouding: List[int] предсказанная персона. массив из 5 элементов,
        где 1 - персона использована, 0 - не использована
    predicted_persona: List[str] предсказанная персона(только использованные)
    predicted_knowledge_index: int предсказанное знание
    predicted_knowledge: str предсказанное знание
    position: int позиция в диалоге
    response: str ответ на вопрос
    """

    knowledge: List[str]
    knowledge_candidates: List[str]
    query: str
    dialog_id: str
    predicted_persona_grouding: List[int]
    predicted_persona: List[str]
    predicted_knowledge_index: int
    predicted_knowledge: str
    position: int
    response: str


class FocusPersonaExtractorDictV1(TypedDict):
    predicted_persona: List[str]
    predicted_persona_grounding: List[int]


class FocusKnowledgeKandidateExtractorV1:
    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        self.model_name = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract(
        self,
        persona: List[str],
        query: str,
        knowledge_candidates: List[str],
    ) -> FocusKnowledgeKandidateExtractorDictV1:
        _persona = " ".join(persona)
        query = query + " " + _persona

        query_emb = self.model.encode([query], convert_to_tensor=True)
        corpus_emb = self.model.encode(knowledge_candidates, convert_to_tensor=True)

        cosine_scores = util.cos_sim(corpus_emb, query_emb)  # type: ignore
        top_indices = cosine_scores.topk(1, dim=0).indices.flatten().tolist()
        top_sentences = [knowledge_candidates[i] for i in top_indices]
        return FocusKnowledgeKandidateExtractorDictV1(
            predicted_index=top_indices[0],
            predicted_knowledge=top_sentences[0],
        )


class FocusPersonaExtractorV1:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-base",
        sample_class=DebertaV3FoCusPersonaTestDatasetSampleDictV2,
        model_sample_class=DebertaV3FoCusPersonaTestDatasetSampleV1,
    ) -> None:
        self.model_name = model_name
        self.model = DebertaV3PersonaClassificationV3.from_pretrained(
            model_name,
            config=DebertaV2Config.from_pretrained(
                model_name,
            ),
        )
        self.model.eval()  # type: ignore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore

        self.hyperparameters = DebertaV3HyperparametersV1(
            train_batch_size=16,
            valid_batch_size=16,
            max_dialog_history_tokens=70,
            max_knowledge_candidates_tokens=220,
            max_persona_tokens=20,
            model_name=model_name,
            project_name="focus_persona_classification",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.sample_class = sample_class
        self.model_sample_class = model_sample_class

    def extract(
        self,
        persona_sentences: List[str],
        used_knowledge: str,
        query: str,
    ) -> FocusPersonaExtractorDictV1:
        model_persona_samples = []
        for persona_sentence in persona_sentences:
            sample = self.sample_class(
                persona_sentence=persona_sentence,
                used_knowledge=used_knowledge,
                query=query,
            )
            model_persona_sample = self.model_sample_class(
                dataset_sample=sample,
                tokenizer=self.tokenizer,
                h_params=self.hyperparameters,
            )
            model_persona_samples.append(model_persona_sample.get_dict())

        predictions = []
        persona_preds: List[str] = []

        for i, model_persona_sample in enumerate(model_persona_samples):
            for key in model_persona_sample.keys():
                model_persona_sample[key] = torch.tensor(model_persona_sample[key])
                model_persona_sample[key] = model_persona_sample[key].unsqueeze(0)

            outputs = self.model(  # type: ignore
                **model_persona_sample,
            )
            logits = outputs.logits
            pred = logits.argmax(dim=1).item()
            if pred == 1:
                persona = persona_sentences[i]
                persona_preds.append(persona)
            predictions.append(pred)

        return FocusPersonaExtractorDictV1(
            predicted_persona=persona_preds,
            predicted_persona_grounding=predictions,
        )


class ResponseGeneratorV1:
    def __init__(
        self,
        model_name: str = "./bart_base_2cx77pua",
    ) -> None:

        self.hyperparameters = BartHyperparametersV3(
            model_name=model_name,
        )

        self.tokenizer = BartFoCusTokenizerV2.from_pretrained(
            self.hyperparameters.model_name,
            hyperparameters=self.hyperparameters,
        )
        self.model = BartLMV7.from_pretrained(
            model_name,
            config=BartConfig.from_pretrained(
                self.hyperparameters.model_name,
            ),  # type: ignore
            hyperparameters=self.hyperparameters,
            tokenizer=self.tokenizer,
        )
        self.model.eval()  # type: ignore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore

    def generate_response(self, res_sample: BartRensponseTestDatasetDictV1) -> str:
        sample = BartFoCusTestDatasetSampleV1(
            focus_dataset_sample=res_sample,
            tokenizer=self.tokenizer,  # type: ignore
            h_params=self.hyperparameters,
        )
        sample = sample.get_dict()
        for key in sample:
            sample[key] = torch.tensor(sample[key])
            sample[key] = sample[key].unsqueeze(0)
            sample[key] = sample[key].to(self.device)

        generated_responses = self.model.generate(  # type: ignore
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            max_length=100,
        )
        generated_responses = self.tokenizer.batch_decode(
            generated_responses,
            skip_special_tokens=True,
        )
        generated_response = generated_responses[0]
        return generated_response


class BartFocusTestDatasetV1:
    def __init__(
        self,
        initial_dataset: FoCusTestDatasetV1,
        knowledge_kandidate_extractor: FocusKnowledgeKandidateExtractorV1,
        focus_persona_extractor: FocusPersonaExtractorV1,
        response_generator: ResponseGeneratorV1,
    ) -> None:
        self.initial_dataset = initial_dataset
        self.knowledge_extractor = knowledge_kandidate_extractor
        self.persona_extractor = focus_persona_extractor
        self.response_generator = response_generator
        self.dataset: List[BartFocusTestDatasetDictV1] = []

        self.dataset = self.__build_dataset()

    def __build_dataset(self) -> List[BartFocusTestDatasetDictV1]:
        dataset = []

        for i, sample in enumerate(self.initial_dataset):  # type: ignore
            persona = sample["persona"]
            query = sample["query"]
            knowledge = sample["knowledge"]
            position = sample["position"]
            knowledge_candidates = sample["knowledge_candidates"]
            knowledge_prediction = self.knowledge_extractor.extract(
                persona=persona,
                query=query,
                knowledge_candidates=knowledge_candidates,
            )
            predicted_knowledge_index = knowledge_prediction["predicted_index"]
            predicted_knowledge = knowledge_prediction["predicted_knowledge"]

            persona_prediction = self.persona_extractor.extract(
                persona_sentences=persona,
                used_knowledge=predicted_knowledge,
                query=query,
            )

            predicted_persona_grounding = persona_prediction[
                "predicted_persona_grounding"
            ]
            predicted_persona = persona_prediction["predicted_persona"]

            response_sample = BartRensponseTestDatasetDictV1(
                persona=predicted_persona,
                query=query,
                knowledge_candidate=predicted_knowledge,
            )

            bot_response = self.response_generator.generate_response(
                res_sample=response_sample,
            )

            dataset_sample = BartFocusTestDatasetDictV1(
                knowledge=knowledge,
                query=query,
                dialog_id=sample["dialog_id"],
                predicted_persona_grouding=predicted_persona_grounding,
                predicted_persona=predicted_persona,
                predicted_knowledge_index=predicted_knowledge_index,
                predicted_knowledge=predicted_knowledge,
                position=position,
                response=bot_response,
                knowledge_candidates=knowledge_candidates,
            )
            dataset.append(dataset_sample)
            print(f"Progress {i}/{len(self.initial_dataset)}")

        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> BartFocusTestDatasetDictV1:
        return self.dataset[index]

    def save_dataset_to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.dataset, f, indent=2)
