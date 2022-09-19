import torch

from train_bart_LM import (
    BartFoCusDatasetSampleHyperparametersV1,
    BartFoCusTokenizerV1,
    BartLMV1,
    FoCusDataModuleV1,
)

from transformers import BartConfig


class Experiment:
    def __init__(
        self,
        model=None,
        data_module=None,
    ):
        self.model = model

        self.data_module = data_module
        train_dataloader = self.data_module.train_dataloader()
        valid_dataloader = self.data_module.val_dataloader()

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
        )

    def train(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()

            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += float(loss.item())
            print(loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        print("Total loss: ", total_loss / len(self.train_dataloader))

    def valid(self):
        self.model.eval()
        total_valid_loss = 0
        for batch in self.valid_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()

            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_valid_loss += float(loss.item())
            print(loss)

        print("Total valid loss: ", total_valid_loss / len(self.valid_dataloader))

    def run_experiment(self, epochs=1):
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            self.train()
            self.valid()


if __name__ == "__main__":

    hyperparameters = BartFoCusDatasetSampleHyperparametersV1()
    tokenizer = BartFoCusTokenizerV1.from_pretrained(
        hyperparameters.bart_model_name, hyperparameters=hyperparameters
    )

    model = BartLMV1(
        config=BartConfig.from_pretrained(hyperparameters.bart_model_name),
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
    )

    model.resize_token_embeddings(len(tokenizer))

    data_module = FoCusDataModuleV1(
        train_path_dataset="./datasets/FoCus/train_focus.json",
        valid_path_dataset="./datasets/FoCus/valid_focus.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
    )
    data_module.setup()

    experiment = Experiment(
        model=model,
        data_module=data_module,
    )
