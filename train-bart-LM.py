from torch.utils.data import Dataset
from typing import List, Dict, TypedDict, Optional
import json
from transformers import AutoTokenizer, BartTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import chain

from datetime import datetime

# import datasets
import torch
from torch import nn
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from pytorch_lightning.loggers import WandbLogger
import os
import wandb
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd

from transformers import BartModel, BartConfig, BartPretrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from pytorch_lightning.loggers import WandbLogger


class TF_IDF:
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


class FoCusTF_IDF(TF_IDF):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.cached_similar = {}

    def top_similar(
        self, query: List[List[int]] = None, top_k: int = 1
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


class BartFoCusDatasetSampleHyperparametersV1:
    def __init__(
        self,
        dialog_history_length: int = 1,
        context_length: int = 1,
        knowledge_length: int = 1,
    ) -> None:
        # fmt: off
        r"""
        Args:
			dialog_history_length (int): количество пар диалогов(назад), которые будут 
				использоваться для генерации ответа
			context_length (int): количество предложений из диалога, относительно которых 
				будут выбираться похожие из поля knowledge
			knowledge_length (int): количество предложений из knowledge, которые будут 
				подаваться на вход модели
        """
        # fmt: on
        self.dialog_history_length = dialog_history_length
        self.context_length = context_length
        self.knowledge_length = knowledge_length

        self.max_persona_tokens = 200
        self.max_dialog_history_tokens = 200
        self.max_knowledge_tokens = 200
        self.max_bot_response_tokens = 150

        self.dialog_bos_token = "<dialog>"
        self.dialog_eos_token = "</dialog>"

        self.seed = 2022
        self.train_batch_size = 4
        self.valid_batch_size = 8

        self.warmup_steps = 10
        self.learning_rate = 6.25e-5
        self.adam_epsilon = 1e-8

        self.gradient_accumulation_steps = 1
        self.train_epochs = 1

        self.bart_model_name = "facebook/bart-base"


class BartFoCusTokenizerV1(BartTokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        *args,
        hyperparameters: BartFoCusDatasetSampleHyperparametersV1 = None,
        **kwargs,
    ):

        tokenizer: BartTokenizer = BartTokenizer.from_pretrained(*args, **kwargs)

        if hyperparameters is not None:
            tokens = [
                hyperparameters.dialog_bos_token,
                hyperparameters.dialog_eos_token,
            ]

            tokenizer.add_special_tokens({"additional_special_tokens": tokens})

        return tokenizer


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
        h_params: BartFoCusDatasetSampleHyperparametersV1 = None,
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
            persona, add_special_tokens=False
        )

        dialog_history = dialog[-2 * dialog_history_length :]
        dialog_history_feature = self.tokenizer.batch_encode_plus(
            dialog_history[:-1], add_special_tokens=False
        )
        dialog_history_target = self.tokenizer.batch_encode_plus(
            dialog_history[-1:], add_special_tokens=False
        )

        # контекст на основе которого подбирается knowledge
        query_context = dialog_history_feature["input_ids"][-context_length:]
        encoded_knowledge = self.tokenizer.batch_encode_plus(
            knowledge, add_special_tokens=False
        )

        tf_idf = FoCusTF_IDF(corpus=encoded_knowledge["input_ids"])
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
        self, initial_dataset: Dict = None
    ) -> List[FoCusDatasetSampleDictV1]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for i, dialog_set in enumerate(initial_dataset_data):
            persona = dialog_set["persona"]
            utterances = dialog_set["utterance"]
            knowledge = dialog_set["knowledge"]

            for j, utterance in enumerate(utterances):
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
        hyperparameters: BartFoCusDatasetSampleHyperparametersV1 = None,
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


class FoCusDataModuleV1(LightningDataModule):
    def __init__(
        self,
        train_path_dataset: str = None,
        valid_path_dataset: str = None,
        hyperparameters: BartFoCusDatasetSampleHyperparametersV1 = None,
        tokenizer: BartFoCusTokenizerV1 = None,
    ) -> None:
        super().__init__()

        self.train_path_dataset = train_path_dataset
        self.valid_path_dataset = valid_path_dataset

        self.train_dataset = None
        self.valid_dataset = None

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

    def setup(self, stage: str = None) -> None:
        train_dataset = FoCusDatasetV1(input_dataset_path=self.train_path_dataset)
        valid_dataset = FoCusDatasetV1(input_dataset_path=self.valid_path_dataset)

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


class BartLMV1(BartPretrainedModel):
    # fmt: off
    r"""
    Simple usage:
		model = BartLMV1(
			config=BartConfig.from_pretrained('facebook/bart-large'),
			hyperparameters=BartFoCusDatasetSampleHyperparametersV1(),
			tokenizer=BartFoCusTokenizerV1.from_pretrained(
				'facebook/bart-base',
				hyperparameters=BartFoCusDatasetSampleHyperparametersV1()),
		)

    	input_ids = torch.tensor([[1, 2, ]])
    	attention_mask = torch.tensor([[1, 1,]])
    	model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			labels=input_ids,
    	)
    """
    # fmt: on

    def __init__(
        self,
        config: BartConfig = None,
        hyperparameters: BartFoCusDatasetSampleHyperparametersV1 = None,
        tokenizer: BartFoCusTokenizerV1 = None,
    ) -> None:
        super().__init__(config=config)
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters

        self.model = BartModel(config=config)
        self.lm_head = nn.Linear(config.d_model, len(tokenizer), bias=False)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Seq2SeqLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits: torch.Tensor = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # copy from https://github.com/pkchat-focus/FoCus/blob/main/classification_modules.py#L462
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            # не очень понимаю что это за ключ в контексте модели BART
            encoder_last_hidden_state=outputs[0],
        )


class BARTModelV1(LightningModule):
    def __init__(
        self,
        hyperparameters: BartFoCusDatasetSampleHyperparametersV1 = None,
        tokenizer: BartFoCusTokenizerV1 = None,
        is_training: bool = False,
    ) -> None:
        super().__init__()
        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.hparams.update(self.hyperparameters.__dict__)

        self.model = BartLMV1(
            config=BartConfig.from_pretrained(hyperparameters.bart_model_name),
            hyperparameters=hyperparameters,
            tokenizer=tokenizer,
        )
        if is_training:
            self.model.resize_token_embeddings(len(tokenizer))

        # self.automatic_optimization = False

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Seq2SeqLMOutput:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.hyperparameters.learning_rate,
            eps=self.hyperparameters.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameters.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: List, batch_idx: int) -> Dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html#id2
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()
        # sch = self.lr_schedulers()
        # sch.step()

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # return {"loss": loss}
        return loss

    def validation_step(self, batch: List, batch_idx: int) -> Dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )


if __name__ == "__main__":
    hyperparameters = BartFoCusDatasetSampleHyperparametersV1()
    tokenizer = BartFoCusTokenizerV1.from_pretrained(
        hyperparameters.bart_model_name, hyperparameters=hyperparameters
    )

    data_module = FoCusDataModuleV1(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
    )
    model = BARTModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        is_training=True,
    )

    wandb_logger = WandbLogger(project="Test", name=hyperparameters.bart_model_name)

    trainer = Trainer(
        max_epochs=hyperparameters.train_epochs,
        accelerator="gpu",
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
