from collections import defaultdict
from typing import Dict

from core.base_models.debertav3_models import (
    DebertaV3ForClassificationV1,
    DebertaV3ForClassificationV2,
    DebertaV3PersonaClassificationV1,
    DebertaV3PersonaClassificationV2,
)
from core.base_models.model_outputs.debertav3_outputs import DebertaV3OutputV1
from core.hyperparameters.debertav3_hyperparameters import DebertaV3HyperparametersV1

from pytorch_lightning import LightningModule

import torch

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.utils.dummy_sentencepiece_objects import DebertaV2Tokenizer


class DebertaV3LightningModelV1(LightningModule):
    def __init__(
        self,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        base_model: DebertaV3ForClassificationV1 | DebertaV3ForClassificationV2,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters(ignore=["base_model"])

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = base_model

        self.predictions_results = {"train": {}, "valid": {}}

    def forward(
        self,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return self.model(**kwargs)

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):

        outputs: DebertaV3OutputV1 = self.model.forward(
            **batch,
        )

        loss = outputs.loss

        task_accuracy = self._compute_batch_metrics(
            mode="train",
            batch=batch,
            outputs=outputs,
        )

        self.log(
            "train_loss",
            loss,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_binary_accuracy",
            task_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def _compute_batch_metrics(
        self,
        mode: str,
        batch: Dict,
        outputs: DebertaV3OutputV1,
    ) -> float:
        """
        Args:
            mode (str): "train" or "valid"
            batch (Dict): dataset batch
            outputs (DebertaV3OutputV1): model outputs
        """
        logits = outputs.logits
        predicts = logits.tolist()
        unique_ids = outputs.unique_ids
        labels = batch["labels"]
        epoch = self.current_epoch
        epoch_key = f"epoch_{epoch}"
        for pred, unique_id, label in zip(predicts, unique_ids, labels):
            if epoch_key not in self.predictions_results[mode]:
                self.predictions_results[mode][epoch_key] = defaultdict(list)
            self.predictions_results[mode][epoch_key][unique_id].append(
                [pred[1], label.item()],
            )

        task_predicts = logits.argmax(dim=-1).view(-1)
        task_labels = labels.view(-1)
        task_accuracy = self._accuracy(task_predicts, task_labels)

        return task_accuracy

    def _accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return (preds == targets).float().mean().cpu().item()

    def validation_step(self, batch, batch_idx: int):

        outputs = self.model.forward(**batch)
        loss = outputs.loss

        task_accuracy = self._compute_batch_metrics(
            mode="valid",
            batch=batch,
            outputs=outputs,
        )

        self.log(
            "valid_loss",
            loss,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "valid_binary_accuracy",
            task_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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
        train_accuracy = self._compute_epoch_accuracy(mode="train")

        self.log(
            "train_epoch_knowledge_accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        valid_accuracy = self._compute_epoch_accuracy(mode="valid")

        self.log(
            "valid_epoch_knowledge_accuracy",
            valid_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def _compute_epoch_accuracy(self, mode: str) -> float:
        current_epoch = self.current_epoch
        epoch_key = f"epoch_{current_epoch}"
        correct_answers = 0
        all_samples_ids = self.predictions_results[mode][epoch_key].keys()
        for pred_id in all_samples_ids:
            predictions = []
            labels = []
            for pred, label in self.predictions_results[mode][epoch_key][pred_id]:
                predictions.append(pred)
                labels.append(label)
            pred_index = torch.tensor(predictions).argmax().item()
            if 1 in labels:
                true_index = labels.index(1)
            else:
                print("No true label")
                print(predictions)
                print(pred_index)
                print(labels)
                print("ERROR" * 1000)
                true_index = 0

            if pred_index == true_index:
                correct_answers += 1

        accuracy = correct_answers / len(all_samples_ids)
        return accuracy


class DebertaV3PersonaLightningModelV1(LightningModule):
    def __init__(
        self,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        base_model: DebertaV3PersonaClassificationV1,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters(ignore=["base_model"])

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = base_model

    def forward(
        self,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return self.model(**kwargs)

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):

        outputs: DebertaV3OutputV1 = self.model.forward(
            **batch,
        )

        loss = outputs.loss

        accuracy = self.compute_accuracy(
            batch,
            outputs,
        )

        self.log(
            "train_loss",
            loss,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def _accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return (preds == targets).float().mean().cpu().item()

    def validation_step(self, batch, batch_idx: int):

        outputs = self.model.forward(**batch)
        loss = outputs.loss
        accuracy = self.compute_accuracy(
            batch,
            outputs,
        )
        self.log(
            "valid_loss",
            loss,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "valid_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
        )

    def compute_accuracy(self, batch, outputs):
        logits = outputs.logits
        labels = batch["labels"]
        task_predicts = logits.argmax(dim=-1).view(-1)
        task_labels = labels.view(-1)
        task_accuracy = self._accuracy(task_predicts, task_labels)
        return task_accuracy

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


class DebertaV3PersonaLightningModelV2(LightningModule):
    def __init__(
        self,
        hyperparameters: DebertaV3HyperparametersV1,
        tokenizer: DebertaV2Tokenizer,
        base_model: DebertaV3PersonaClassificationV2,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters(ignore=["base_model"])

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = base_model

        self.predictions_results = {"train": {}, "valid": {}}

    def forward(
        self,
        **kwargs,
    ) -> DebertaV3OutputV1:
        return self.model(**kwargs)

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):

        outputs: DebertaV3OutputV1 = self.model.forward(
            **batch,
        )

        loss = outputs.loss

        task_accuracy = self._compute_batch_metrics(
            mode="train",
            batch=batch,
            outputs=outputs,
        )

        self.log(
            "train_loss",
            loss,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_binary_accuracy",
            task_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def _compute_batch_metrics(
        self,
        mode: str,
        batch: Dict,
        outputs: DebertaV3OutputV1,
    ) -> float:
        """
        Args:
            mode (str): "train" or "valid"
            batch (Dict): dataset batch
            outputs (DebertaV3OutputV1): model outputs
        """
        logits = outputs.logits
        predicts = (torch.sigmoid(logits) > 0.5).int().flatten()
        unique_ids = outputs.unique_ids
        labels = batch["labels"]
        epoch = self.current_epoch
        epoch_key = f"epoch_{epoch}"
        for pred, unique_id, label in zip(predicts, unique_ids, labels):
            if epoch_key not in self.predictions_results[mode]:
                self.predictions_results[mode][epoch_key] = defaultdict(list)
            self.predictions_results[mode][epoch_key][unique_id].append(
                [pred.item(), label.item()],
            )

        task_accuracy = self._accuracy(predicts, labels)

        return task_accuracy

    def _accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return (preds == targets).float().mean().cpu().item()

    def validation_step(self, batch, batch_idx: int):

        outputs = self.model.forward(**batch)
        loss = outputs.loss

        task_accuracy = self._compute_batch_metrics(
            mode="valid",
            batch=batch,
            outputs=outputs,
        )

        self.log(
            "valid_loss",
            loss,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "valid_binary_accuracy",
            task_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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
        train_accuracy = self._compute_epoch_accuracy(mode="train")

        self.log(
            "train_epoch_persona_accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        valid_accuracy = self._compute_epoch_accuracy(mode="valid")

        self.log(
            "valid_epoch_persona_accuracy",
            valid_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def _compute_epoch_accuracy(self, mode: str) -> float:
        current_epoch = self.current_epoch
        epoch_key = f"epoch_{current_epoch}"
        correct_answers = 0
        all_samples_ids = self.predictions_results[mode][epoch_key].keys()

        for pred_id in all_samples_ids:
            predictions = []
            labels = []
            for pred, label in self.predictions_results[mode][epoch_key][pred_id]:
                predictions.append(pred)
                labels.append(label)
            acc = self._accuracy(
                torch.tensor(predictions),
                torch.tensor(labels),
            )

            correct_answers += acc

        accuracy = correct_answers / len(all_samples_ids)
        return accuracy
