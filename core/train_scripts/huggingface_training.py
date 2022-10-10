from core.dataloaders.focus.focus_dataloader import FoCusDatasetPersonaV2
from core.dataloaders.focus.models.debertav3_dataloaders import (
    DebertaV3FoCusPersonaDatasetSampleV2,
)
from core.hyperparameters.debertav3_hyperparameters import DebertaV3HyperparametersV1
from core.utils import PytorchDatasetFactory

from datasets import load_metric  # type: ignore

import numpy as np

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from transformers import DataCollatorWithPadding  # type: ignore

import time

if __name__ == "__main__":
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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

    training_args = TrainingArguments(
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

    trainer = Trainer(
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
