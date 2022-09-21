from typing import Dict, List

from core.base_models.bart_models import BartLMV1, BartLMV2
from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV1,
    BartHyperparametersV2,
)
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1

from pytorch_lightning import LightningModule

import torch

from transformers import BartConfig  # type: ignore
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.optimization import get_linear_schedule_with_warmup


class BARTLightningModelV1(LightningModule):
    def __init__(
        self,
        hyperparameters: BartHyperparametersV1,
        tokenizer: BartFoCusTokenizerV1,
        is_training: bool = False,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters()

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = BartLMV1(
            config=BartConfig.from_pretrained(hyperparameters.model_name),  # type: ignore
            hyperparameters=hyperparameters,
            tokenizer=tokenizer,
        )
        if is_training:
            self.model.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: List, batch_idx: int):
        input_ids = batch["input_ids"]  # type: ignore
        attention_mask = batch["attention_mask"]  # type: ignore
        labels = input_ids.clone()

        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        self.log(
            "train_loss",
            loss,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # return {"loss": loss}
        return loss

    def validation_step(self, batch: List, batch_idx: int):
        input_ids = batch["input_ids"]  # type: ignore
        attention_mask = batch["attention_mask"]  # type: ignore
        labels = input_ids.clone()

        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self.log(
            "valid_loss",
            loss,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
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


class BARTLightningModelV2(LightningModule):
    def __init__(
        self,
        hyperparameters: BartHyperparametersV2,
        tokenizer: BartFoCusTokenizerV1,
        is_training: bool = False,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters()

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = BartLMV2(
            config=BartConfig.from_pretrained(hyperparameters.model_name),  # type: ignore
            hyperparameters=hyperparameters,
            tokenizer=tokenizer,
        )
        if is_training:
            self.model.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: Dict, batch_idx: int):

        outputs = self.model.forward(
            **batch,
        )

        loss = outputs.loss

        self.log(
            "train_loss",
            loss,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):

        outputs = self.model.forward(**batch)
        loss = outputs.loss
        self.log(
            "valid_loss",
            loss,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
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
