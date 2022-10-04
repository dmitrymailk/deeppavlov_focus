from dataclasses import dataclass
from typing import Dict


@dataclass
class DebertaV3HyperparametersV1:
    dialog_history_length: int = 1
    knowledge_length: int = 1
    # examples lengths
    max_persona_tokens: int = 15
    max_dialog_history_tokens: int = 80
    max_knowledge_candidates_tokens: int = 250
    max_full_dialog_history_tokens: int = 2 * 80
    max_full_knowledge_candidates_tokens: int = 3 * 250
    # tokens
    seed: int = 2022
    train_batch_size: int = 4
    valid_batch_size: int = 4
    warmup_steps: int = 100
    learning_rate: float = 6.25e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    train_epochs: int = 1
    model_name: str = "microsoft/deberta-v3-base"
    project_name: str = "focus_knowledge_classification"
    lighting_hyperparameters: Dict | None = None
    """
        максимум может быть 512 токенов
        данные делятся на:
        в каждом идет речь об одном примере из подгруппы датасета
        - dialog history. оптимально обрезать до 80 токенов
        - knowledge_candidates. оптимально обрезать до 250 токенов

        """
