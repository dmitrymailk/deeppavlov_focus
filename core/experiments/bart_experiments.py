from core.base_models.bart_models import BartLMV3, BartLMV4
from core.dataloaders.focus_dataloader import (
    FoCusLightningDataModuleV1,
    FoCusLightningDataModuleV2,
    FoCusLightningDataModuleV3,
)
from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV1,
    BartHyperparametersV2,
    BartHyperparametersV3,
)
from core.hyperparameters.lighting_hyperparameters import LightingHyperparametersV1
from core.lighting_models.bart_lighting import (
    BARTLightningModelV1,
    BARTLightningModelV2,
)
from core.loggers.wandb_logger import WandbLoggerV1
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV1, BartFoCusTokenizerV2
from core.utils import ExperimentArgumentParserV1, TrainArgumentsV1

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import BartConfig  # type: ignore


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

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
    ).__dict__

    hyperparameters = BartHyperparametersV2(
        gradient_accumulation_steps=3,
        lighting_hyperparameters=lighting_hyperparameters,
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
        **lighting_hyperparameters,
    )

    trainer.fit(model, datamodule=data_module)


def experiment_v3() -> None:
    """
    BART с измененным loss

    loss = LM_loss
    в этом эксперименте мы не используем loss из persona и knowledge
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
    ).__dict__

    hyperparameters = BartHyperparametersV2(
        gradient_accumulation_steps=3,
        lighting_hyperparameters=lighting_hyperparameters,
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
    base_model = BartLMV3(
        config=BartConfig.from_pretrained(hyperparameters.model_name),  # type: ignore
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
    )
    model = BARTLightningModelV2(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_training=True,
        base_model=base_model,
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
        **lighting_hyperparameters,
    )

    trainer.fit(model, datamodule=data_module)


def experiment_v4() -> None:
    """
    [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query><response>[dialog][-1]</response>[EOS]
    [persona] - склееенные предложения персоны, 5шт
    query - это последний вопрос от пользователя
    response - это ответ от бота за запрос пользователя
    [knowledge_candidates] - это топ 2 похожих предложений из
        knowledge_candidates на query

    классификацию knowledge_candidates на основе:
        - <query>
        - </query>
        - [EOS]
        - [SEP] после [knowledge_candidates]
        - [BOS]

    классификацию persona на основе:
        - <query>
        - </query>
        - [EOS]
        - [SEP] после [persona]
        - [BOS]

    Bart с loss = lm_loss + knowledge_candidates_loss + persona_loss
    отличие от v2 в том что я взял другие фичи для классификации
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=1,
    ).__dict__

    hyperparameters = BartHyperparametersV3(
        lighting_hyperparameters=lighting_hyperparameters,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV2.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.is_debug

    data_module = FoCusLightningDataModuleV3(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_debug=is_debug,
    )
    base_model = BartLMV4(
        config=BartConfig.from_pretrained(hyperparameters.model_name),  # type: ignore
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
    )
    # model = BARTLightningModelV2(
    #     hyperparameters=hyperparameters,
    #     tokenizer=tokenizer,  # type: ignore
    #     is_training=True,
    #     base_model=base_model,
    # )
    model = BARTLightningModelV2.load_from_checkpoint(
        "./Test/3ntglw7k/checkpoints/facebook/bart-base-epoch=00-val_loss=0.00.ckpt",
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV1(
        hyperparameters=hyperparameters,
        is_debug=True,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{valid_loss:.2f}",
    )

    trainer = Trainer(
        accelerator="gpu",
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
        **lighting_hyperparameters,
    )

    trainer.fit(model, datamodule=data_module)
