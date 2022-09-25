from dataclasses import dataclass
from typing import Dict


@dataclass
class BartHyperparametersV1:
    dialog_history_length: int = 1
    context_length: int = 1
    knowledge_length: int = 1
    max_persona_tokens: int = 200
    max_dialog_history_tokens: int = 200
    max_knowledge_tokens: int = 200
    max_bot_response_tokens: int = 150
    dialog_bos_token: str = "<dialog>"
    dialog_eos_token: str = "</dialog>"
    seed: int = 2022
    train_batch_size: int = 4
    valid_batch_size: int = 4
    warmup_steps: int = 100
    learning_rate: float = 6.25e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    train_epochs: int = 1
    model_name: str = "facebook/bart-base"


@dataclass
class BartHyperparametersV2:
    dialog_history_length: int = 1
    context_length: int = 1
    knowledge_length: int = 1
    max_persona_tokens: int = 200
    max_dialog_history_tokens: int = 200
    max_knowledge_tokens: int = 200
    max_bot_response_tokens: int = 150
    dialog_bos_token: str = "<dialog>"
    dialog_eos_token: str = "</dialog>"
    seed: int = 2022
    train_batch_size: int = 4
    valid_batch_size: int = 4
    warmup_steps: int = 100
    learning_rate: float = 6.25e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    train_epochs: int = 1
    model_name: str = "facebook/bart-base"
    persona_labels_amount: int = 5
    knowledge_labels_amount: int = 10
    lighting_hyperparameters: Dict | None = None


@dataclass
class BartHyperparametersV3:
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
    seed: int = 2022
    train_batch_size: int = 4
    valid_batch_size: int = 4
    warmup_steps: int = 100
    learning_rate: float = 6.25e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    train_epochs: int = 1
    model_name: str = "facebook/bart-base"
    persona_labels_amount: int = 5
    knowledge_labels_amount: int = 10
    lighting_hyperparameters: Dict | None = None
    """
        максимум может быть 1024 токена
        данные делятся на:
        в каждом идет речь об одном примере из подгруппы датасета
        - persona. оптимально обрезать до 15 токенов
        - dialog history. оптимально обрезать до 80 токенов
        - knowledge. оптимально обрезать до 420 токенов
        - knowledge_candidates. оптимально обрезать до 250 токенов

        в идеале должно быть так:
        у персоны 5 предложений => persona=5*20=100
        у диалога 2 предложений => dialog=2*80=160
        у knowledge 2 предложения => knowledge=2*420=840
        у knowledge_candidates 10 предложений => knowledge_candidates=10*250=2500
        """
