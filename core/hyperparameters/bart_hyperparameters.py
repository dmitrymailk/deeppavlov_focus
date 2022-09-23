from typing import Dict


class BartHyperparametersV1:
    def __init__(
        self,
        dialog_history_length: int = 1,
        context_length: int = 1,
        knowledge_length: int = 1,
        max_persona_tokens: int = 200,
        max_dialog_history_tokens: int = 200,
        max_knowledge_tokens: int = 200,
        max_bot_response_tokens: int = 150,
        dialog_bos_token: str = "<dialog>",
        dialog_eos_token: str = "</dialog>",
        seed: int = 2022,
        train_batch_size: int = 4,
        valid_batch_size: int = 4,
        warmup_steps: int = 100,
        learning_rate: float = 6.25e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        train_epochs: int = 1,
        model_name: str = "facebook/bart-base",
    ) -> None:

        self.dialog_history_length = dialog_history_length
        self.context_length = context_length
        self.knowledge_length = knowledge_length

        self.max_persona_tokens = max_persona_tokens
        self.max_dialog_history_tokens = max_dialog_history_tokens
        self.max_knowledge_tokens = max_knowledge_tokens
        self.max_bot_response_tokens = max_bot_response_tokens

        self.dialog_bos_token = dialog_bos_token
        self.dialog_eos_token = dialog_eos_token

        self.seed = seed
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_epochs = train_epochs

        self.model_name = model_name


class BartHyperparametersV2:
    def __init__(
        self,
        dialog_history_length: int = 1,
        context_length: int = 1,
        knowledge_length: int = 1,
        max_persona_tokens: int = 200,
        max_dialog_history_tokens: int = 200,
        max_knowledge_tokens: int = 200,
        max_bot_response_tokens: int = 150,
        dialog_bos_token: str = "<dialog>",
        dialog_eos_token: str = "</dialog>",
        seed: int = 2022,
        train_batch_size: int = 4,
        valid_batch_size: int = 4,
        warmup_steps: int = 100,
        learning_rate: float = 6.25e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        train_epochs: int = 1,
        model_name: str = "facebook/bart-base",
        persona_labels_amount: int = 5,
        knowledge_labels_amount: int = 10,
        lighting_hyperparameters: Dict = {},
    ) -> None:

        self.dialog_history_length = dialog_history_length
        self.context_length = context_length
        self.knowledge_length = knowledge_length

        self.max_persona_tokens = max_persona_tokens
        self.max_dialog_history_tokens = max_dialog_history_tokens
        self.max_knowledge_tokens = max_knowledge_tokens
        self.max_bot_response_tokens = max_bot_response_tokens

        self.dialog_bos_token = dialog_bos_token
        self.dialog_eos_token = dialog_eos_token

        self.seed = seed
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_epochs = train_epochs

        self.model_name = model_name

        self.persona_labels_amount = persona_labels_amount
        self.knowledge_labels_amount = knowledge_labels_amount
        self.lighting_hyperparameters = lighting_hyperparameters


class BartHyperparametersV3:
    def __init__(
        self,
        dialog_history_length: int = 1,
        context_length: int = 1,
        knowledge_length: int = 1,
        # examples lengths
        max_persona_tokens: int = 20,
        max_dialog_history_tokens: int = 80,
        max_knowledge_tokens: int = 420,
        max_knowledge_candidates_tokens: int = 250,
        max_full_persona_tokens: int = 5 * 20,
        max_full_dialog_history_tokens: int = 2 * 80,
        max_full_knowledge_tokens: int = 420,
        max_full_knowledge_candidates_tokens: int = 3 * 250,
        # tokens
        response_bos_token: str = "<response>",
        response_eos_token: str = "</response>",
        query_bos_token: str = "<query>",
        query_eos_token: str = "</query>",
        seed: int = 2022,
        train_batch_size: int = 4,
        valid_batch_size: int = 4,
        warmup_steps: int = 100,
        learning_rate: float = 6.25e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        train_epochs: int = 1,
        model_name: str = "facebook/bart-base",
        persona_labels_amount: int = 5,
        knowledge_labels_amount: int = 10,
        lighting_hyperparameters: Dict = {},
    ) -> None:

        self.dialog_history_length = dialog_history_length
        self.context_length = context_length
        self.knowledge_length = knowledge_length
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
        # examples lengths
        self.max_persona_tokens = max_persona_tokens
        self.max_dialog_history_tokens = max_dialog_history_tokens
        self.max_knowledge_tokens = max_knowledge_tokens
        self.max_knowledge_candidates_tokens = max_knowledge_candidates_tokens
        self.max_full_persona_tokens = max_full_persona_tokens
        self.max_full_dialog_history_tokens = max_full_dialog_history_tokens
        self.max_full_knowledge_tokens = max_full_knowledge_tokens
        self.max_full_knowledge_candidates_tokens = max_full_knowledge_candidates_tokens
        # tokens
        self.response_bos_token = response_bos_token
        self.response_eos_token = response_eos_token
        self.query_bos_token = query_bos_token
        self.query_eos_token = query_eos_token

        self.seed = seed
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_epochs = train_epochs

        self.model_name = model_name

        self.persona_labels_amount = persona_labels_amount
        self.knowledge_labels_amount = knowledge_labels_amount
        self.lighting_hyperparameters = lighting_hyperparameters
