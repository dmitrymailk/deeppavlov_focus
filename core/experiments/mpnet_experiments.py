import time

from pytorch_lightning import seed_everything

from core.base_models.mpnet_models import (
    MPNetForSequenceClassificationV1,
    MPNetForSequenceClassificationV2,
    MPNetForSentenceEmbeddingV1,
    MPNetForSentenceEmbeddingV2,
)
from core.base_models.debertav3_models import DebertaV3ForSentenceEmbeddingV1
from core.dataloaders.focus.focus_dataloader import (
    FoCusDatasetKnowledgeV3,
    FoCusDatasetKnowledgeV4,
    FoCusDatasetKnowledgeV5,
    FoCusDatasetPersonaV2,
)
from core.lighting_models.mpnet_lighting import (
    MPNetKnowledgeLightningModelV1,
    MPNetKnowledgeLightningModelV2,
)
from core.dataloaders.focus.models.mpnet_dataloaders import (
    MPNetFoCusKnowledgeDatasetSampleV2,
    MPNetFoCusPersonaDatasetSampleV1,
)
from core.hyperparameters.lighting_hyperparameters import LightingHyperparametersV1
from core.hyperparameters.mpnet_hyperparameters import MPNetHyperparametersV1
from core.loggers.wandb_logger import WandbLoggerV2
from core.utils import (
    ExperimentArgumentParserV1,
    PytorchDatasetFactory,
    TrainArgumentsV1,
)

from datasets import load_metric  # type: ignore

import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import transformers as tr


from core.dataloaders.focus.lighting.mpnet_lighting_dataloader import (
    MPNetLightingDataModuleV1,
    MPNetLightingDataModuleV2,
)
from core.dataloaders.focus.models.mpnet_dataloaders import (
    MPNetFoCusKnowledgeDatasetSampleV1,
)


def experiment_1() -> None:
    """
    использую взвешенный лосс
    """
    hyperparameters = MPNetHyperparametersV1(
        train_batch_size=16,
        valid_batch_size=16,
        max_dialog_history_tokens=80,
        max_knowledge_candidates_tokens=250,
        max_persona_tokens=20,
        model_name="sentence-transformers/all-mpnet-base-v2",
        project_name="focus_persona_classification",
    )
    tokenizer = tr.AutoTokenizer.from_pretrained(  # type: ignore
        hyperparameters.model_name,
    )
    data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)  # type: ignore

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
        dataset_sample_class=MPNetFoCusPersonaDatasetSampleV1,
    )

    valid_dataset = PytorchDatasetFactory(
        dataset=valid_dataset,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        dataset_sample_class=MPNetFoCusPersonaDatasetSampleV1,
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
    class_weights = [0.13454188704999148, 0.8654581129500085]
    print("Class weights: ", class_weights)

    class_weights = torch.tensor(class_weights)

    model = MPNetForSequenceClassificationV1.from_pretrained(  # type: ignore
        "sentence-transformers/all-mpnet-base-v2",
        cross_entropy_loss_weights=class_weights,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # type: ignore

    training_args = tr.TrainingArguments(  # type: ignore
        output_dir=f"./results/{hyperparameters.model_name}",
        learning_rate=hyperparameters.learning_rate,
        per_device_train_batch_size=hyperparameters.train_batch_size,
        per_device_eval_batch_size=hyperparameters.valid_batch_size,
        num_train_epochs=4,
        weight_decay=0.00001,
        logging_steps=10,
        overwrite_output_dir=True,
        run_name=f"huggingface_{hyperparameters.model_name}",
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=3000,
        save_steps=3000,
        do_train=True,
        load_best_model_at_end=True,
        warmup_steps=hyperparameters.warmup_steps,
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
    trainer.save_model(f"./results/{hyperparameters.model_name}_{time_string}")


def experiment_2() -> None:
    """
    использую дефолтный focal loss
    """
    hyperparameters = MPNetHyperparametersV1(
        train_batch_size=16,
        valid_batch_size=16,
        max_dialog_history_tokens=80,
        max_knowledge_candidates_tokens=250,
        max_persona_tokens=20,
        model_name="sentence-transformers/all-mpnet-base-v2",
        project_name="focus_persona_classification",
    )
    tokenizer = tr.AutoTokenizer.from_pretrained(  # type: ignore
        hyperparameters.model_name,
    )
    data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)  # type: ignore

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
        dataset_sample_class=MPNetFoCusPersonaDatasetSampleV1,
    )

    valid_dataset = PytorchDatasetFactory(
        dataset=valid_dataset,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        dataset_sample_class=MPNetFoCusPersonaDatasetSampleV1,
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
    class_weights = [0.13454188704999148, 0.8654581129500085]
    print("Class weights: ", class_weights)

    class_weights = torch.tensor(class_weights)

    model = MPNetForSequenceClassificationV2.from_pretrained(  # type: ignore
        "sentence-transformers/all-mpnet-base-v2",
        cross_entropy_loss_weights=class_weights,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # type: ignore

    training_args = tr.TrainingArguments(  # type: ignore
        output_dir=f"./results/{hyperparameters.model_name}",
        learning_rate=hyperparameters.learning_rate,
        per_device_train_batch_size=hyperparameters.train_batch_size,
        per_device_eval_batch_size=hyperparameters.valid_batch_size,
        num_train_epochs=4,
        weight_decay=0.00001,
        logging_steps=10,
        overwrite_output_dir=True,
        run_name=f"huggingface_{hyperparameters.model_name}",
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=3000,
        save_steps=3000,
        do_train=True,
        # load_best_model_at_end=True,
        warmup_steps=hyperparameters.warmup_steps,
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
    run_id = wandb.run.id  # type: ignore
    trainer.save_model(f"./results/saved/{hyperparameters.model_name}/{run_id}")


def experiment_4() -> None:
    """
    экперименты с symmetric crosse entropy loss
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 4
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        max_epochs=max_epochs,
        deterministic=False,
    ).__dict__

    hyperparameters = MPNetHyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        project_name="focus_knowledge_classification",
        train_batch_size=16,
        valid_batch_size=16,
        # model_name="sentence-transformers/all-mpnet-base-v2",
        model_name="microsoft/deberta-v3-small",
    )
    seed_everything(hyperparameters.seed)

    tokenizer = tr.AutoTokenizer.from_pretrained(hyperparameters.model_name)  # type: ignore
    tokenizer.model_max_length = 512
    is_debug = args.debug_status

    data_module = MPNetLightingDataModuleV1(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
        base_train_dataset_class=FoCusDatasetKnowledgeV4,
        base_valid_dataset_class=FoCusDatasetKnowledgeV3,
        base_train_sample_class=MPNetFoCusKnowledgeDatasetSampleV1,
        base_valid_sample_class=MPNetFoCusKnowledgeDatasetSampleV1,
    )

    # base_model = MPNetForSentenceEmbeddingV1.from_pretrained(hyperparameters.model_name)
    # base_model = MPNetForSentenceEmbeddingV2.from_pretrained(hyperparameters.model_name)
    base_model = DebertaV3ForSentenceEmbeddingV1.from_pretrained(
        hyperparameters.model_name,
    )

    model = MPNetKnowledgeLightningModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,  # type: ignore
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_accuracy",
        mode="max",
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{valid_accuracy:.2f}",
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
    trainer.validate(model=model, dataloaders=data_module)
    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path=ckpt_path,
    )


def experiment_5() -> None:
    """
    пробую самописный triplet loss
    1 - source * positive + source * negative
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 4
    if args.debug_status == 1:
        max_epochs = 1

    lighting_hyperparameters = LightingHyperparametersV1(
        precision=16,
        # accumulate_grad_batches=3,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = MPNetHyperparametersV1(
        lighting_hyperparameters=lighting_hyperparameters,
        project_name="focus_knowledge_classification",
        train_batch_size=8,
        valid_batch_size=32,
        model_name="sentence-transformers/all-mpnet-base-v2",
    )
    seed_everything(hyperparameters.seed)

    tokenizer = tr.AutoTokenizer.from_pretrained(hyperparameters.model_name)  # type: ignore
    tokenizer.model_max_length = 512
    is_debug = args.debug_status

    data_module = MPNetLightingDataModuleV2(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        debug_status=is_debug,
        base_train_dataset_class=FoCusDatasetKnowledgeV5,
        base_valid_dataset_class=FoCusDatasetKnowledgeV3,
        base_train_sample_class=MPNetFoCusKnowledgeDatasetSampleV2,
        base_valid_sample_class=MPNetFoCusKnowledgeDatasetSampleV1,
    )

    base_model = MPNetForSentenceEmbeddingV1.from_pretrained(hyperparameters.model_name)
    # base_model = MPNetForSentenceEmbeddingV2.from_pretrained(hyperparameters.model_name)
    # base_model = DebertaV3ForSentenceEmbeddingV1.from_pretrained(
    #     hyperparameters.model_name,
    # )

    model = MPNetKnowledgeLightningModelV2(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,  # type: ignore
        base_model=base_model,  # type: ignore
    )

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_accuracy",
        mode="max",
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{valid_accuracy:.2f}",
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
    if args.debug_status != 1:
        trainer.validate(model=model, dataloaders=data_module)

    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path=ckpt_path,
    )
