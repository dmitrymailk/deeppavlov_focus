import argparse
from dataclasses import dataclass
from typing import List

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
            token_pattern=r"(?u)\b\d+\b",
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
    ) -> List[List[int]]:
        query = self.__encode_sentences(query)
        query = self.vectorizer.transform(query)

        similarity = cosine_similarity(self.X, query)
        similarity = similarity.flatten()
        similarity = np.argsort(similarity)[::-1][:top_k]
        similarity = similarity.tolist()

        similar_samples = [self.corpus[i] for i in similarity]
        return similar_samples


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
    ) -> List[List[int]]:
        query_str = str(query)

        if query_str in self.cached_similar:
            return self.cached_similar[query_str]

        similar_samples = super().top_similar(
            query=query,
            top_k=top_k,
        )
        self.cached_similar[query_str] = similar_samples

        return similar_samples


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
