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

        """
        Args:
            dialog_history_length (int): количество пар диалогов(назад),
                которые будут использоваться для генерации ответа
            context_length (int): количество предложений из диалога, относительно которых
                будут выбираться похожие из поля knowledge
            knowledge_length (int): количество предложений из knowledge, которые будут
                подаваться на вход модели
        """
        # fmt: on
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

        """
        Args:
            dialog_history_length (int): количество пар диалогов(назад),
                которые будут использоваться для генерации ответа
            context_length (int): количество предложений из диалога, относительно которых
                будут выбираться похожие из поля knowledge
            knowledge_length (int): количество предложений из knowledge, которые будут
                подаваться на вход модели
        """
        # fmt: on
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
