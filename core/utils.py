import argparse
from dataclasses import dataclass
from itertools import chain
from typing import List, TypeVar

from datasets import load_metric  # type: ignore

import numpy as np

from rouge_score import rouge_scorer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset

from torchmetrics import CHRFScore  # type: ignore


class TfIdf:
    def __init__(
        self,
        corpus: List[List[int]],
    ) -> None:
        # fmt: off
        """
        Example usage:
                corpus = [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                ]

                tf_idf = TF_IDF(corpus)
                similar_sentences = tf_idf.get_similar([1, 2, 3], n=3)
                >>> similar_sentences
                [
                    [1, 2, 3],
                    [1, 2, 3, 4],
                    [1, 2]
                ]

        Args:
                corpus (List[List[int]]): токенизированный корпус
        """
        # fmt: on

        self.vectorizer = TfidfVectorizer(
            # token_pattern is number
            token_pattern=r"\b\d+\b",
        )
        new_corpus = self.__encode_sentences(corpus)

        self.X = self.vectorizer.fit_transform(new_corpus)
        self.corpus = corpus

    def __encode_sentence(self, sentence: List[int]) -> str:
        return " ".join(list(map(str, sentence)))

    def __encode_sentences(self, sentences: List[List[int]]) -> List[str]:
        return list(map(self.__encode_sentence, sentences))

    def top_similar(
        self,
        query: List[List[int]] = None,
        top_k: int = 1,
        return_indices: bool = False,
    ) -> List[List[int]]:
        query = self.__encode_sentences(query)  # type: ignore
        query = self.vectorizer.transform(query)

        similarity = cosine_similarity(self.X, query)
        similarity = similarity.flatten()
        similarity = np.argsort(similarity)[::-1][:top_k]
        similarity = similarity.tolist()
        if not return_indices:
            similar_samples = [self.corpus[i] for i in similarity]
            return similar_samples

        return similarity


class FoCusTfIdf(TfIdf):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.cached_similar = {}

    def top_similar(
        self,
        query: List[List[int]] = None,
        top_k: int = 1,
        return_indices: bool = False,
    ) -> List[List[int]]:
        query_str = str(query)

        if query_str in self.cached_similar:
            return self.cached_similar[query_str]

        similar_samples = super().top_similar(
            query=query,
            top_k=top_k,
            return_indices=return_indices,
        )
        self.cached_similar[query_str] = similar_samples

        return similar_samples


class TextEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.bleu = load_metric("sacrebleu")
        self.chrf = CHRFScore()

    def evaluate(self, generated_texts: List[str], original_texts: List[str]):
        blue_score = self.bleu.compute(
            predictions=generated_texts,  # type: ignore
            references=[[item] for item in original_texts],
        )["score"]

        # compute rouge score
        rougeL_score = 0
        for gen_text, orig_text in zip(generated_texts, original_texts):
            scores = self.rouge.score(orig_text, gen_text)
            rougeL_score += scores["rougeL"].fmeasure

        rougeL_score /= len(generated_texts)

        # compute chrf score
        chrf_score = self.chrf(
            generated_texts, [[item] for item in original_texts]
        ).item()

        return {
            "blue_score": blue_score,
            "rougeL_score": rougeL_score,
            "chrf_score": chrf_score,
        }


@dataclass
class TrainArgumentsV1:
    debug_status: int


class ExperimentArgumentParserV1:
    """Todo: сделать типизацию через наследование от Namespace"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="training arguments")
        params = [
            (
                "--debug_status",
                {
                    "dest": "debug_status",
                    "type": int,
                    "default": 0,
                },
            ),
        ]

        for name, param in params:
            parser.add_argument(name, **param)

        args = parser.parse_args()
        args = args._get_kwargs()
        args = {arg[0]: arg[1] for arg in args}

        args = TrainArgumentsV1(**args)

        self.args = args


def flat_list(list_of_lists: List[List] | List) -> List:
    return list(chain.from_iterable(list_of_lists))


InitialDatasetClass = TypeVar("InitialDatasetClass")
TokenizerClass = TypeVar("TokenizerClass")
DatasetSampleClass = TypeVar("DatasetSampleClass")
HyperParametersClass = TypeVar("HyperParametersClass")


class PytorchDatasetFactory:
    def __init__(
        self,
        dataset: InitialDatasetClass,
        tokenizer: TokenizerClass,
        dataset_sample_class: DatasetSampleClass,
        hyperparameters: HyperParametersClass,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.dataset_sample_class = dataset_sample_class
        self.hyperparameters = hyperparameters

    def __get_fabric(self) -> Dataset:
        parent_self = self

        class GeneralDataset(Dataset):
            def __init__(
                self,
                dataset: InitialDatasetClass,
                tokenizer: TokenizerClass,
                hyperparameters: HyperParametersClass,
            ) -> None:
                self.dataset = dataset
                self.tokenizer = tokenizer
                self.hyperparameters = hyperparameters

            def __len__(self) -> int:
                return len(self.dataset)  # type: ignore

            def __getitem__(self, idx: int) -> DatasetSampleClass:
                return parent_self.dataset_sample_class(  # type: ignore
                    self.dataset[idx],  # type: ignore
                    self.tokenizer,  # type: ignore
                    self.hyperparameters,
                ).get_dict()  # type: ignore

        dataset = GeneralDataset(
            self.dataset,
            self.tokenizer,
            self.hyperparameters,
        )
        return dataset

    def __new__(cls, *args, **kwargs) -> Dataset:
        new_object = super(PytorchDatasetFactory, cls).__new__(
            cls,
        )
        new_object.__init__(*args, **kwargs)
        return new_object.__get_fabric()


def experiment_decorator(function):
    def wrapper():
        doc = function.__doc__
        if doc is None:
            doc = "No description"

        func_name = function.__name__
        doc = f"{func_name}: {doc}"

        result = function(doc)
        return result

    return wrapper
