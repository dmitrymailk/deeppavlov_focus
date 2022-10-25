from core.base_models.debertav3_models import (
    DebertaV3ForClassificationV1,
    DebertaV3ForClassificationV2,
    DebertaV3PersonaClassificationV1,
    DebertaV3PersonaClassificationV2,
)
from core.dataloaders.focus.lighting.debertav3_lighting_dataloaders import (
    DebertaV3FoCusLightningDataModuleV1,
    DebertaV3FoCusLightningDataModuleV2,
    DebertaV3FoCusLightningDataModuleV3,
    DebertaV3FoCusLightningDataModuleV4,
    DebertaV3FoCusLightningDataModuleV5,
    DebertaV3FoCusLightningDataModuleV6,
    DebertaV3FoCusPersonaLightningDataModuleV1,
    DebertaV3FoCusPersonaLightningDataModuleV2,
)
from core.hyperparameters.debertav3_hyperparameters import DebertaV3HyperparametersV1
from core.hyperparameters.lighting_hyperparameters import LightingHyperparametersV1
from core.lighting_models.debertav3_lighting import (
    DebertaV3LightningModelV1,
    DebertaV3PersonaLightningModelV1,
    DebertaV3PersonaLightningModelV2,
)
from core.loggers.wandb_logger import WandbLoggerV2
from core.utils import (
    ExperimentArgumentParserV1,
    PytorchDatasetFactory,
    TrainArgumentsV1,
    experiment_decorator,
)
from core.dataloaders.focus.focus_dataloader import FoCusDatasetPersonaV2
from core.dataloaders.focus.models.debertav3_dataloaders import (
    DebertaV3FoCusPersonaDatasetSampleV2,
)


import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import DebertaV2Config, DebertaV2Tokenizer  # type: ignore


from datasets import load_metric  # type: ignore

import numpy as np

import transformers as tr

import time

from core.base_models.debertav3_models import DebertaV3PersonaClassificationV3
import torch


def experiment_1():
    """
    бинарная классифицикация
    для этого я использую последний вопрос пользователя и кандидатов
    из knowledge_candidates. где предложению которое использовалось в
    ответе соответсвует 1, а остальным 0. для группировки по диалогам пришлось испльзовать
    уникальные id чтобы потом посчитать accuracy сравнимую с результатами модели bart.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 1
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )
    is_debug = args.debug_status

    data_module = DebertaV3FoCusLightningDataModuleV1(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    trainer = pl.Trainer(
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


def experiment_2():
    """
    похоже на experiment_1, теперь количество положительных примеров
    равно количеству отрицательных примеров.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 1
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )
    is_debug = args.debug_status

    data_module = DebertaV3FoCusLightningDataModuleV2(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    trainer = pl.Trainer(
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


def experiment_3():
    """
    похоже на experiment_1, теперь количество положительных примеров
    равно количеству отрицательных примеров.
    увеличил количество эпох. увеличил количество batch gradient accumulation.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 3
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=4,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
        experiment_description=experiment_3.__doc__,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV2(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
        **lighting_hyperparameters,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


def experiment_4():
    """
    похоже на experiment_3, количество положительных примеров
    равно количеству отрицательных примеров.
    добавил в текст используемые части персоны(если такие использовались)
    для генерации ответа
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV3(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


def experiment_5():
    """
    похоже на experiment_4.
    только использую microsoft/deberta-v3-large
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=2,
        valid_batch_size=4,
        model_name="microsoft/deberta-v3-large",
        experiment_description=__doc__,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV3(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


def experiment_6():
    """
    похоже на experiment_3.
    только использую microsoft/deberta-v3-large
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=16,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=2,
        valid_batch_size=4,
        model_name="microsoft/deberta-v3-large",
        experiment_description=experiment_6.__doc__,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV2(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


@experiment_decorator
def experiment_7(doc: str = ""):
    """
    использую microsoft/deberta-v3-base
    увеличил контекст. теперь
    буду брать последний вопрос от пользователя, предыдущий ответ бота и предыдущий
    вопрос пользователя. не использую персону.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
        model_name="microsoft/deberta-v3-base",
        experiment_description=doc,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV4(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1269dck1/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.50.ckpt"  # noqa: E501

    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
        **lighting_hyperparameters,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


@experiment_decorator
def experiment_8(doc: str = ""):
    """
    использую microsoft/deberta-v3-base
    увеличил контекст. теперь
    буду брать последний вопрос от пользователя, предыдущий ответ бота и предыдущий
    вопрос пользователя. не использую персону.
    убрал context pooler и dropout
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
        model_name="microsoft/deberta-v3-base",
        experiment_description=doc,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV4(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV2(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


@experiment_decorator
def experiment_9(doc: str = ""):
    """
    использую microsoft/deberta-v3-base
    беру последний вопрос от пользователя, предыдущий ответ бота и предыдущий
    вопрос пользователя. использую персону.
    немного изменил параметры обрезания последовательностей.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
        max_dialog_history_tokens=70,
        max_knowledge_candidates_tokens=220,
        max_persona_tokens=15,
        model_name="microsoft/deberta-v3-base",
        experiment_description=doc,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV5(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


@experiment_decorator
def experiment_10(doc: str = ""):
    """
    использую microsoft/deberta-v3-base
    беру последний вопрос от пользователя, предыдущий ответ бота и предыдущий
    вопрос пользователя. использую персону.
    немного изменил параметры обрезания последовательностей.
    в отличие от experiment_9 использую другой датасет.
    теперь я буду использовать все примеры из датасета.
    хоть и сделаю тем самым его имбалансным.
    будет всего 1 положительный пример на 9 отрицательных.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
        max_dialog_history_tokens=70,
        max_knowledge_candidates_tokens=220,
        max_persona_tokens=15,
        model_name="microsoft/deberta-v3-base",
        experiment_description=doc,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusLightningDataModuleV6(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3ForClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3LightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


@experiment_decorator
def experiment_11(doc: str = ""):
    """
    использую microsoft/deberta-v3-base
    классификация персоны.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    is_debug = args.debug_status

    max_epochs = 2
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=1,
        valid_batch_size=4,
        # max_dialog_history_tokens=70,
        # max_knowledge_candidates_tokens=220,
        # max_persona_tokens=15,
        model_name="microsoft/deberta-v3-base",
        experiment_description=doc,
        project_name="focus_persona_classification",
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusPersonaLightningDataModuleV1(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = DebertaV3PersonaClassificationV1(
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
    )
    model = DebertaV3PersonaLightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


@experiment_decorator
def experiment_12(doc: str = ""):
    """
    использую "microsoft/deberta-v3-small"
    бинарная классификация персоны.
    сбалансированный датасет.
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args
    debug_status = args.debug_status

    max_epochs = 100
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
    ).__dict__

    hyperparameters = DebertaV3HyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=16,
        valid_batch_size=16,
        # max_dialog_history_tokens=70,
        # max_knowledge_candidates_tokens=220,
        # max_persona_tokens=15,
        model_name="microsoft/deberta-v3-small",
        experiment_description=doc,
        project_name="focus_persona_classification",
    )
    seed_everything(hyperparameters.seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        hyperparameters.model_name,
    )

    data_module = DebertaV3FoCusPersonaLightningDataModuleV2(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=debug_status,
    )
    # base_model = DebertaV3PersonaClassificationV2(
    #     config=DebertaV2Config.from_pretrained(
    #         hyperparameters.model_name,
    #     ),  # type: ignore
    # )

    base_model = DebertaV3PersonaClassificationV2.from_pretrained(
        hyperparameters.model_name,
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
        ),
    )

    model = DebertaV3PersonaLightningModelV2(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,  # type: ignore
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
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

    # ckpt_path = "/home/dimweb/Desktop/deeppavlov/my_focus/focus_knowledge_classification/1is9z2lu/checkpoints/microsoft/deberta-v3-base-epoch=00-valid_loss=0.53.ckpt"  # noqa: E501

    trainer = pl.Trainer(
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


@experiment_decorator
def experiment_13(doc: str = ""):
    model_name = "microsoft/deberta-v3-small"
    tokenizer = tr.AutoTokenizer.from_pretrained(model_name)  # type: ignore
    data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)  # type: ignore
    model = tr.AutoModelForSequenceClassification.from_pretrained(  # type: ignore
        model_name,
        num_labels=2,
    )

    hyperparameters = DebertaV3HyperparametersV1(
        train_batch_size=16,
        valid_batch_size=16,
        max_dialog_history_tokens=70,
        max_knowledge_candidates_tokens=220,
        max_persona_tokens=20,
        model_name=model_name,
        project_name="focus_persona_classification",
    )

    train_dataset = FoCusDatasetPersonaV2(
        input_dataset_path="./datasets/FoCus/train_focus.json",
        is_train=True,
    )

    valid_dataset = FoCusDatasetPersonaV2(
        input_dataset_path="./datasets/FoCus/valid_focus.json",
        is_train=False,
    )

    train_dataset = PytorchDatasetFactory(
        dataset=train_dataset,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        dataset_sample_class=DebertaV3FoCusPersonaDatasetSampleV2,
    )

    valid_dataset = PytorchDatasetFactory(
        dataset=valid_dataset,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        dataset_sample_class=DebertaV3FoCusPersonaDatasetSampleV2,
    )

    accuracy_metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    training_args = tr.TrainingArguments(  # type: ignore
        output_dir=f"./results/{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.015,
        logging_steps=10,
        overwrite_output_dir=True,
        run_name=f"huggingface_{model_name}",
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=3000,
        save_steps=3000,
        do_train=True,
        load_best_model_at_end=True,
    )

    trainer = tr.Trainer(  # type: ignore
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=valid_dataset,  # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
    )

    trainer.train()
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    trainer.save_model(f"./results/{model_name}_{time_string}")


def experiment_14(doc: str = ""):
    """
    использую взвешенный лосс
    """
    model_name = "microsoft/deberta-v3-small"
    tokenizer = tr.AutoTokenizer.from_pretrained(model_name)  # type: ignore
    data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)  # type: ignore

    hyperparameters = DebertaV3HyperparametersV1(
        train_batch_size=16,
        valid_batch_size=16,
        max_dialog_history_tokens=80,
        max_knowledge_candidates_tokens=250,
        max_persona_tokens=20,
        model_name=model_name,
        project_name="focus_persona_classification",
    )

    train_dataset = FoCusDatasetPersonaV2(
        input_dataset_path="./datasets/FoCus/train_focus.json",
        is_train=False,
    )

    valid_dataset = FoCusDatasetPersonaV2(
        input_dataset_path="./datasets/FoCus/valid_focus.json",
        is_train=False,
    )

    train_dataset = PytorchDatasetFactory(
        dataset=train_dataset,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        dataset_sample_class=DebertaV3FoCusPersonaDatasetSampleV2,
    )

    valid_dataset = PytorchDatasetFactory(
        dataset=valid_dataset,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        dataset_sample_class=DebertaV3FoCusPersonaDatasetSampleV2,
    )

    accuracy_metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # train_positive = 0
    # train_negative = 0
    # for sample in train_dataset:  # type: ignore
    #     if sample["labels"] == 1:
    #         train_positive += 1
    #     else:
    #         train_negative += 1

    # print("Train positive: ", train_positive)
    # print("Train negative: ", train_negative)
    # print("Train ratio: ", train_positive / (train_positive + train_negative))

    # positive_ratio = train_positive / (train_positive + train_negative)
    # Class weights:  [0.13454188704999148, 0.8654581129500085]
    # class_weights = [positive_ratio, 1 - positive_ratio]
    class_weights = [0.1, 0.9]
    print("Class weights: ", class_weights)

    class_weights = torch.tensor(class_weights)

    model = DebertaV3PersonaClassificationV3.from_pretrained(  # type: ignore
        hyperparameters.model_name,
        config=DebertaV2Config.from_pretrained(
            hyperparameters.model_name,
            hidden_dropout_prob=0.3,
            pooler_dropout=0.1,
        ),
        class_weights=class_weights,
    )

    training_args = tr.TrainingArguments(  # type: ignore
        output_dir=f"./results/{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=hyperparameters.train_batch_size,
        per_device_eval_batch_size=hyperparameters.valid_batch_size,
        num_train_epochs=4,
        weight_decay=0.00001,
        logging_steps=10,
        overwrite_output_dir=True,
        run_name=f"huggingface_{model_name}",
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=3000,
        save_steps=3000,
        do_train=True,
        load_best_model_at_end=True,
    )

    trainer = tr.Trainer(  # type: ignore
        model=model,  # type: ignore
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=valid_dataset,  # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
    )

    trainer.train()
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    trainer.save_model(f"./results/{model_name}_{time_string}")
