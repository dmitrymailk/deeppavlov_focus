from core.base_models.bart_models import (  # noqa: F401
    BartLMV7,
    BartLMV8,
)
from core.dataloaders.focus_dataloader import (
    FoCusLightningDataModuleV3,
    FoCusLightningDataModuleV4,
)
from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV3,
)
from core.hyperparameters.lighting_hyperparameters import LightingHyperparametersV1
from core.lighting_models.bart_lighting import (
    BARTLightningModelV2,
)
from core.loggers.wandb_logger import WandbLoggerV1
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV2
from core.utils import ExperimentArgumentParserV1, TrainArgumentsV1

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import BartConfig  # type: ignore


def experiment_v4() -> None:
    """
    seq2seq model with BartForConditionalGeneration
    input_ids:
        [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query>[EOS]
    labels:
        [BOS]<response>[dialog][-1]</response>[EOS]

    Модель у которой следующий лосс
    loss = loss_LM + loss_persona + loss_knowledge_candidates
    где
        loss_LM - лосс языковой модели
        loss_persona - лосс при классификации persona
        loss_knowledge_candidates - лосс при классификации knowledge candidates

    классификацию persona на основе:
        - <query>
        - </query>
        - [EOS]
        - [SEP] после [persona]
        - [BOS]
    классификацию knowledge_candidates на основе:
        - <query>
        - </query>
        - [EOS]
        - [SEP] после [knowledge_candidates]
        - [BOS]
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=2,
    ).__dict__

    hyperparameters = BartHyperparametersV3(
        lighting_hyperparameters=lighting_hyperparameters,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV2.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.debug_status

    data_module = FoCusLightningDataModuleV3(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = BartLMV7(
        config=BartConfig.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
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
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{valid_loss:.2f}",
    )

    accelerator = "gpu"
    if args.debug_status == 1:
        accelerator = "cpu"

    trainer = Trainer(
        accelerator=accelerator,
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
        **lighting_hyperparameters,
    )

    trainer.fit(model, datamodule=data_module)


def experiment_v5() -> None:
    """
    seq2seq model with BartForConditionalGeneration
    input_ids:
        [BOS][persona][SEP][knowledge_candidate][SEP]<query>[dialog][-2]</query>[EOS]
    labels:
        [BOS]<response>[dialog][-1]</response>[EOS]

    Модель у которой следующий лосс
    loss = loss_LM + loss_persona + loss_knowledge_candidates
    где
        loss_LM - лосс языковой модели
        loss_persona - лосс при классификации persona
        loss_knowledge_candidates - лосс при классификации knowledge candidates

    классификацию persona на основе:
        - <query>
        - </query>
        - [EOS]
        - [SEP] после [persona]
        - [BOS]
    классификацию knowledge_candidates на основе:
        - <query>
        - </query>
        - [EOS]
        - [SEP] после [knowledge_candidates]
        - [BOS]

    в этом эксперименте в качестве персоны используются только те предложения
    которые использовались для генерации ответа. тоже самое и с knowledge_candidates.
    этот эксперимет нужен чтобы проверить идеальные условия для модели.
    если бы мы скажем сделали точный экстрактор персоны и knowledge_candidates, который бы
    выдавал именно те предложения которые использовались для генерации ответа
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 4
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = BartHyperparametersV3(
        lighting_hyperparameters=lighting_hyperparameters,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV2.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.debug_status

    data_module = FoCusLightningDataModuleV4(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = BartLMV7(
        config=BartConfig.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
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
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{valid_loss:.2f}",
    )

    accelerator = "gpu"
    if args.debug_status == 1:
        accelerator = "cpu"

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/Test/1z9mgq52/checkpoints/facebook/bart-base-epoch=01-valid_loss=4.49.ckpt"  # noqa: E501

    trainer = Trainer(
        accelerator=accelerator,
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
        **lighting_hyperparameters,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path=ckpt_path,
    )


def experiment_v6() -> None:
    """
    seq2seq model with BartForConditionalGeneration
    тоже самое что и experiment_v5, но теперь используется только loss языкового
    моделирования
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 4
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = BartHyperparametersV3(
        lighting_hyperparameters=lighting_hyperparameters,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV2.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.debug_status

    data_module = FoCusLightningDataModuleV4(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = BartLMV8(
        config=BartConfig.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
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
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{valid_loss:.2f}",
    )

    accelerator = "gpu"
    if args.debug_status == 1:
        accelerator = "cpu"

    # ckpt_path = ""  # noqa: E501

    trainer = Trainer(
        accelerator=accelerator,
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
        **lighting_hyperparameters,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path=ckpt_path,
    )
