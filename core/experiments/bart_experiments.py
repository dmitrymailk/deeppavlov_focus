from core.dataloaders.focus_dataloader import (
    FoCusLightningDataModuleV1,
    FoCusLightningDataModuleV2,
)
from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV1,
    BartHyperparametersV2,
)
from core.lighting_models.bart_lighting import (
    BARTLightningModelV1,
    BARTLightningModelV2,
)
from core.loggers.wandb_logger import WandbLoggerV1
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1
from core.utils import ExperimentArgumentParserV1, TrainArgumentsV1

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


def experiment_v1() -> None:
    """
    простейший бейзлайн на BART для языкового моделирования
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    hyperparameters = BartHyperparametersV1(
        gradient_accumulation_steps=3,
    )

    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV1.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.is_debug

    data_module = FoCusLightningDataModuleV1(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_debug=is_debug,
    )
    model = BARTLightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_training=True,
    )

    wandb_logger = WandbLoggerV1(
        hyperparameters=hyperparameters,
        is_debug=True,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        max_epochs=hyperparameters.train_epochs,
        accelerator="gpu",
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)


def experiment_v2() -> None:
    """
    BART с измененным loss

    loss = LM_loss + persona_loss + knowledge_loss
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    hyperparameters = BartHyperparametersV2(
        gradient_accumulation_steps=3,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV1.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.is_debug

    data_module = FoCusLightningDataModuleV2(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_debug=is_debug,
    )
    model = BARTLightningModelV2(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_training=True,
    )

    wandb_logger = WandbLoggerV1(
        hyperparameters=hyperparameters,
        is_debug=True,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        max_epochs=hyperparameters.train_epochs,
        accelerator="gpu",
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)
