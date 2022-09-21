from typing import List

from core.base_models.bart_models import BartLMV1, BartLMV2, BartLMV2Outputs, BartLMV3
from core.dataloaders.focus_dataloader import FoCusLightningDataModuleV2DictV1
from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV1,
    BartHyperparametersV2,
)
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1

from pytorch_lightning import LightningModule

import torch
from torch.nn.functional import sigmoid

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
        base_model: BartLMV2 | BartLMV3,
        is_training: bool = False,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters(ignore=["base_model"])

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = base_model
        if is_training:
            self.model.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        **kwargs,
    ) -> BartLMV2Outputs:
        return self.model(**kwargs)

    def training_step(self, batch: FoCusLightningDataModuleV2DictV1, batch_idx: int):

        outputs: BartLMV2Outputs = self.model.forward(
            **batch,
        )

        loss = outputs.loss

        lm_loss = outputs.lm_loss
        persona_loss = outputs.persona_loss
        knowledge_loss = outputs.knowledge_loss

        persona_accuracy = self._compute_persona_accuracy(
            outputs=outputs,
            batch=batch,
        )

        knowledge_accuracy = self._compute_knowledge_accuracy(
            outputs=outputs,
            batch=batch,
        )

        self.log(
            "train_loss",
            loss,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log_dict(
            {
                "train_lm_loss": lm_loss,  # type: ignore
                "train_persona_loss": persona_loss,
                "train_knowledge_loss": knowledge_loss,
            },
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        self.log_dict(
            {
                "train_persona_accuracy": persona_accuracy,
                "train_knowledge_accuracy": knowledge_accuracy,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return loss

    def _accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return (preds == targets).float().mean().cpu().item()

    def _compute_persona_accuracy(
        self,
        outputs: BartLMV2Outputs,
        batch: FoCusLightningDataModuleV2DictV1,
    ) -> float:
        logits = outputs.persona_logits
        targets = batch["persona_grounding"]
        preds = (sigmoid(logits) > 0.5).int().view(-1)
        targets = targets.view(-1)
        return self._accuracy(preds, targets)

    def _compute_knowledge_accuracy(
        self,
        outputs: BartLMV2Outputs,
        batch: FoCusLightningDataModuleV2DictV1,
    ) -> float:
        logits = outputs.knowledge_logits
        targets = batch["knowledge_answer_index"]
        preds = logits.argmax(dim=1).view(-1)
        targets = targets.view(-1)
        return self._accuracy(preds, targets)

    def validation_step(self, batch: FoCusLightningDataModuleV2DictV1, batch_idx: int):

        outputs = self.model.forward(**batch)
        loss = outputs.loss

        lm_loss = outputs.lm_loss
        persona_loss = outputs.persona_loss
        knowledge_loss = outputs.knowledge_loss

        persona_accuracy = self._compute_persona_accuracy(
            outputs=outputs,
            batch=batch,
        )

        knowledge_accuracy = self._compute_knowledge_accuracy(
            outputs=outputs,
            batch=batch,
        )

        self.log(
            "valid_loss",
            loss,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log_dict(
            {
                "valid_lm_loss": lm_loss,  # type: ignore
                "valid_persona_loss": persona_loss,
                "valid_knowledge_loss": knowledge_loss,
            },
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        self.log_dict(
            {
                "valid_persona_accuracy": persona_accuracy,
                "valid_knowledge_accuracy": knowledge_accuracy,
            },
            on_step=False,
            on_epoch=True,
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
