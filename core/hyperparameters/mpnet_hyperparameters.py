from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MPNetHyperparametersV1:
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
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    project_name: str = "focus_persona_classification"
    lighting_hyperparameters: Dict | None = None
    experiment_description: str | None = ""
    class_weights: List[float] | None = None
    """
    максимум может быть 768 токенов
    """
