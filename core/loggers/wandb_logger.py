from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV1,
    BartHyperparametersV2,
)

from pytorch_lightning.loggers import WandbLogger  # type: ignore


class WandbLoggerV1:
    def __init__(
        self,
        hyperparameters: BartHyperparametersV1 | BartHyperparametersV2,
        is_debug: bool = False,
    ) -> None:
        self.is_debug = is_debug
        self.hyperparameters = hyperparameters

    @property
    def logger(self) -> WandbLogger:
        if self.is_debug:
            project = "Test"
        else:
            project = "Test"

        return WandbLogger(
            project=project,
            name=self.hyperparameters.model_name,
        )
