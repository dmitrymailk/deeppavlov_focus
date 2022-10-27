from core.base_models.bart_models import (  # noqa: F401
    BartLMV10,
    BartLMV11,
    BartLMV12,
    BartLMV13,
    BartLMV7,
    BartLMV8,
    BartLMV9,
)
from core.dataloaders.focus.lighting.bart_lighting_dataloaders import (
    FoCusLightningDataModuleV3,
    FoCusLightningDataModuleV4,
    FoCusLightningDataModuleV5,
)
from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV3,
)
from core.hyperparameters.lighting_hyperparameters import LightingHyperparametersV1
from core.lighting_models.bart_lighting import (
    BARTLightningModelV2,
    BARTLightningModelV3,
    BARTLightningModelV4,
)
from core.loggers.wandb_logger import WandbLoggerV1
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV2
from core.utils import ExperimentArgumentParserV1, TrainArgumentsV1

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import torch

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


def experiment_v7() -> None:
    """
    seq2seq model with BartForConditionalGeneration
    тоже самое что и experiment_v6, но увеличил train_batch_size до 8
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
        train_batch_size=8,
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


def experiment_v8() -> None:
    """
    похож на experiment_v4, только тут я используют только knowledge_loss
    seq2seq model with BartForConditionalGeneration
    input_ids:
        [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query>[EOS]
    labels:
        [BOS]<response>[dialog][-1]</response>[EOS]

    Модель у которой следующий лосс
    loss = loss_knowledge_candidates
    где
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
    base_model = BartLMV9(
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


def experiment_v9() -> None:
    """
    input_ids (List[int]):
        [BOS][persona][[knowledge]<query>[dialog][-2]</query>[EOS]
        persona - это сконкатенированная персона
        knowledge - наиболее похожее предложение из базы знаний
            на query
        query - это вопрос, который задал пользователь

    labels (List[int]):
        [BOS]<response>[dialog][-1]</response>[EOS]

    knowledge_candidate_ids (List[int]):
        [BOS][knowledge_candidate][EOS]

    knowledge_ids (List[int]):
        [BOS][knowledge][EOS]

    persona_ids (List[int]):
        [BOS][persona][EOS]

    классификацию knowledge_candidates на основе:
        - [EOS] из knowledge_candidates
        - [EOS] из knowledge
        - [EOS] из persona
        - [EOS] из query

    классификацию persona на основе:
        - [EOS] из knowledge
        - [EOS] из persona
        - [EOS] из query
    теперь я извелекаю фичи не из одной последовательности, а
    из отдельных.
    таким образом не нужно составлять длинные последовательности
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=2,
    ).__dict__

    hyperparameters = BartHyperparametersV3(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=2,
        valid_batch_size=2,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV2.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.debug_status

    data_module = FoCusLightningDataModuleV5(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = BartLMV10(
        config=BartConfig.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
    )
    model = BARTLightningModelV3(
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


def experiment_v10() -> None:
    """
    похож на experiment_v9
    только теперь я увеличил контекст знаний из базы
    и теперь обучаю только классификатор на knowledge_candidates
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        max_epochs=2,
    ).__dict__

    hyperparameters = BartHyperparametersV3(
        lighting_hyperparameters=lighting_hyperparameters,
        train_batch_size=2,
        valid_batch_size=2,
    )
    seed_everything(hyperparameters.seed)

    tokenizer = BartFoCusTokenizerV2.from_pretrained(
        hyperparameters.model_name,
        hyperparameters=hyperparameters,
    )
    is_debug = args.debug_status

    data_module = FoCusLightningDataModuleV5(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
    )
    base_model = BartLMV11(
        config=BartConfig.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
    )
    model = BARTLightningModelV3(
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


def experiment_v11() -> None:
    """
    seq2seq model with BartForConditionalGeneration
    тоже самое что и experiment_v5, но теперь используется только loss языкового
    моделирования и weighted loss
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
    # weights = torch.load(
    #     "/home/dimweb/Desktop/deeppavlov/my_focus/bart_tokens_statistics.pt",
    # )
    # weights = 1 - weights.to("cuda")
    # weights = torch.ones(weights.shape).to("cuda")

    base_model = BartLMV12.from_pretrained(
        hyperparameters.model_name,
        config=BartConfig.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        weights=None,
    )
    model = BARTLightningModelV4(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_training=True,
        base_model=base_model,  # type: ignore
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


def experiment_v12() -> None:
    """
    seq2seq model with BartForConditionalGeneration
    тоже самое что и experiment_v5, но теперь используется только loss языкового
    моделирования и дефолтный focal loss(без игнонирования паддинга)
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

    base_model = BartLMV13.from_pretrained(
        hyperparameters.model_name,
        config=BartConfig.from_pretrained(
            hyperparameters.model_name,
        ),  # type: ignore
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
    )
    model = BARTLightningModelV4(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        is_training=True,
        base_model=base_model,  # type: ignore
    )

    wandb_logger = WandbLoggerV1(
        hyperparameters=hyperparameters,
        is_debug=True,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_blue_score",
        mode="max",
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
