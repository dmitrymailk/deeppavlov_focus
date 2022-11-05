from core.hyperparameters.mpnet_hyperparameters import MPNetHyperparametersV1
from core.base_models.mpnet_models import MPNetForSentenceEmbeddingV1

from pytorch_lightning import LightningModule

from transformers import AutoTokenizer  # type: ignore
from transformers.optimization import get_linear_schedule_with_warmup

import torch
from torch import nn


class MPNetKnowledgeLightningModelV1(LightningModule):
    def __init__(
        self,
        hyperparameters: MPNetHyperparametersV1,
        tokenizer: AutoTokenizer,
        base_model: MPNetForSentenceEmbeddingV1,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters(ignore=["base_model"])

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = base_model

        self.predicts = {
            "train": {},
            "valid": {},
        }

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):

        sentence_1 = self.model(
            input_ids=batch["sentence_1"],
            attention_mask=batch["attention_mask_1"],
        )

        sentence_2 = self.model(
            input_ids=batch["sentence_2"],
            attention_mask=batch["attention_mask_2"],
        )

        loss_fnc = nn.CrossEntropyLoss()
        labels = torch.tensor(range(len(sentence_1))).to(self.device)
        scores = torch.mm(sentence_1, sentence_2.T)
        loss = (loss_fnc(scores, labels) + loss_fnc(scores.T, labels)) / 2

        self.log("train_loss", loss)

        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        sentence_1 = self.model(
            input_ids=batch["sentence_1"],
            attention_mask=batch["attention_mask_1"],
        )

        sentence_2 = self.model(
            input_ids=batch["sentence_2"],
            attention_mask=batch["attention_mask_2"],
        )

        similarities = sentence_1 * sentence_2
        similarities = similarities.sum(axis=1).tolist()

        for i, utterance_id in enumerate(batch["utterance_id"]):
            score = batch["score"][i]
            if self.predicts["valid"].get(utterance_id) is None:
                self.predicts["valid"][utterance_id] = [[similarities[i], score]]
            else:
                self.predicts["valid"][utterance_id].append([similarities[i], score])

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyperparameters.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyperparameters.weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hyperparameters.learning_rate,
            eps=self.hyperparameters.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameters.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self) -> None:
        correct = 0
        for _, scores in self.predicts["valid"].items():
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            if scores[0][1] == 1:
                correct += 1

        accuracy = correct / len(self.predicts["valid"])
        self.log("valid_accuracy", accuracy)
