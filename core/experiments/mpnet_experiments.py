import time

from core.base_models.mpnet_models import MPNetForSequenceClassificationV1
from core.dataloaders.focus.focus_dataloader import FoCusDatasetPersonaV2
from core.dataloaders.focus.models.mpnet_dataloaders import (
    MPNetFoCusPersonaDatasetSampleV1,
)
from core.hyperparameters.mpnet_hyperparameters import MPNetHyperparametersV1
from core.utils import PytorchDatasetFactory

from datasets import load_metric  # type: ignore

import numpy as np

import torch


import transformers as tr


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
        weight_decay=0.002,
        logging_steps=10,
        overwrite_output_dir=True,
        run_name=f"huggingface_{hyperparameters.model_name}",
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
    trainer.save_model(f"./results/{hyperparameters.model_name}_{time_string}")
