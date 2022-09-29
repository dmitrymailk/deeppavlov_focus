from dataclasses import dataclass
from typing import Dict


@dataclass
class T5HyperparametersV1:
    dialog_history_length: int = 1
    context_length: int = 1
    knowledge_length: int = 1
    # examples lengths
    max_persona_tokens: int = 20
    max_dialog_history_tokens: int = 80
    max_knowledge_tokens: int = 420
    max_knowledge_candidates_tokens: int = 250
    max_full_persona_tokens: int = 5 * 20
    max_full_dialog_history_tokens: int = 2 * 80
    max_full_knowledge_tokens: int = 420
    max_full_knowledge_candidates_tokens: int = 3 * 250
    # tokens
    response_bos_token: str = "<response>"
    response_eos_token: str = "</response>"
    query_bos_token: str = "<query>"
    query_eos_token: str = "</query>"
    sep_token: str = "<sep>"
    seed: int = 2022
    train_batch_size: int = 4
    valid_batch_size: int = 4
    warmup_steps: int = 100
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    model_name: str = "t5-base"
    tokenizer_max_length: int = 512
    persona_labels_amount: int = 5
    knowledge_labels_amount: int = 10
    lighting_hyperparameters: Dict | None = None
    """
    максимум может быть 512 токена
    """
