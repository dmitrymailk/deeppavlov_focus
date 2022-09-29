from typing import List, Optional

from core.base_models.model_outputs.bart_outputs import BartOutputV1
from core.hyperparameters.bart_hyperparameters import (
    BartHyperparametersV3,
)
from core.tokenizers.bart_tokenizers import BartFoCusTokenizerV2

import torch
from torch import nn

# from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartModel,
    # BartPretrainedModel,
)

# from transformers.utils import ModelOutput  # type: ignore


from transformers.models.bart.modeling_bart import shift_tokens_right


class BartLMV7(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(
        self,
        config: BartConfig,
        hyperparameters: BartHyperparametersV3,
        tokenizer: BartFoCusTokenizerV2,
    ):
        super().__init__(config)
        self.model = BartModel(config)
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters
        self.register_buffer("final_logits_bias", torch.zeros((1, len(tokenizer))))
        self.lm_head = nn.Linear(
            config.d_model,
            len(tokenizer),
            bias=False,
        )
        self.persona_head = nn.Linear(
            config.d_model,
            hyperparameters.persona_labels_amount,
            bias=False,
        )
        self.knowledge_candidates_head = nn.Linear(
            config.d_model,
            hyperparameters.knowledge_labels_amount,
            bias=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        # default fields
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # custom fields
        persona_sep_index: Optional[torch.Tensor] = None,
        query_eos_index: Optional[torch.Tensor] = None,
        query_bos_index: Optional[torch.Tensor] = None,
        bos_index: Optional[torch.Tensor] = None,
        eos_index: Optional[torch.Tensor] = None,
        persona_grounding: Optional[torch.Tensor] = None,
        knowledge_answer_index: Optional[torch.Tensor] = None,
        knowledge_candidates_sep_index: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BartOutputV1:
        """
        input_ids:
            [BOS][persona][SEP][knowledge_candidates][SEP]<query>[dialog][-2]</query>[EOS]
        labels:
            [BOS]<response>[dialog][-1]</response>[EOS]

        Модель у которой следующий лосс
        loss = loss_LM + loss_persona + loss_knowledge_candidates
        где
            loss_LM - лосс языковой модели
            loss_persona - лосс при классификации persona
            loss_knowledge_candidates - лосс при классификации knowledge candidates

        классификацию persona на основе:
            - <query>
            - </query>
            - [EOS]
            - [SEP] после [persona]
            - [BOS]
        классификацию knowledge_candidates на основе:
            - <query>
            - </query>
            - [EOS]
            - [SEP] после [knowledge_candidates]
            - [BOS]
        """
        loss: torch.Tensor = torch.tensor(
            0,
            device=self.device,
            dtype=torch.float,
        )
        persona_loss = 0
        knowledge_loss = 0
        lm_loss = None
        persona_logits = None
        knowledge_logits = None

        if labels is not None:

            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels,  # type: ignore
                    self.config.pad_token_id,  # type: ignore
                    self.config.decoder_start_token_id,  # type: ignore
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=labels_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_decoder_states = outputs[0]
        encoder_features = outputs.encoder_last_hidden_state
        lm_logits = self.lm_head(last_decoder_states) + self.final_logits_bias

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,  # type: ignore
            )
            lm_loss = loss_fct(
                lm_logits.view(-1, len(self.tokenizer)),
                labels.view(-1),
            )
            loss += lm_loss

        # extract persona vectors
        # <query></query>[EOS][SEP_persona][BOS]
        if persona_sep_index is not None:
            assert query_eos_index is not None
            assert query_bos_index is not None
            assert bos_index is not None
            assert eos_index is not None

            persona_feature_vectors = []
            for (
                persona_sep_index_i,
                batch_item,
                query_eos_i,
                query_bos_i,
                bos_i,
                eos_i,
            ) in zip(
                persona_sep_index,
                encoder_features,
                query_eos_index,
                query_bos_index,
                bos_index,
                eos_index,
            ):
                persona_sep_vector = batch_item[persona_sep_index_i]
                query_eos_vector = batch_item[query_eos_i]
                query_bos_vector = batch_item[query_bos_i]
                bos_vector = batch_item[bos_i]
                eos_vector = batch_item[eos_i]

                persona_sep_vector += (
                    query_eos_vector + query_bos_vector + bos_vector + eos_vector
                )
                persona_feature_vectors.append(persona_sep_vector)

            persona_vector = torch.vstack(persona_feature_vectors)
            persona_logits = self.persona_head(persona_vector)

            # compute persona loss
            if persona_grounding is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                persona_grounding = persona_grounding.type_as(persona_logits)
                persona_loss = loss_fct(persona_logits, persona_grounding)
                loss += persona_loss

        if knowledge_candidates_sep_index is not None:
            # extract knowledge vectors
            # <query></query>[EOS][SEP_knowledge_candidates][BOS]
            assert query_eos_index is not None
            assert query_bos_index is not None
            assert bos_index is not None
            assert eos_index is not None
            knowledge_candidates_feature_vectors = []
            for (
                knowledge_sep_index_i,
                batch_item,
                query_bos_i,
                query_eos_i,
                bos_i,
                eos_i,
            ) in zip(
                knowledge_candidates_sep_index,
                encoder_features,
                query_bos_index,
                query_eos_index,
                bos_index,
                eos_index,
            ):
                knowledge_sep_vector = batch_item[knowledge_sep_index_i]
                query_eos_vector = batch_item[query_eos_i]
                query_bos_vector = batch_item[query_bos_i]
                bos_vector = batch_item[bos_i]
                eos_vector = batch_item[eos_i]

                knowledge_sep_vector += (
                    query_eos_vector + query_bos_vector + bos_vector + eos_vector
                )
                knowledge_candidates_feature_vectors.append(
                    knowledge_sep_vector,
                )

            knowledge_vector = torch.vstack(knowledge_candidates_feature_vectors)
            knowledge_logits = self.knowledge_candidates_head(knowledge_vector)

            # compute knowledge loss
            if knowledge_answer_index is not None:
                loss_fct = nn.CrossEntropyLoss()
                knowledge_loss = loss_fct(
                    knowledge_logits,
                    knowledge_answer_index.view(-1),
                )
                loss += knowledge_loss

        return BartOutputV1(
            # default fields
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            # custom fields
            lm_loss=lm_loss,
            persona_loss=persona_loss,
            knowledge_loss=knowledge_loss,
            persona_logits=persona_logits,
            knowledge_logits=knowledge_logits,
            last_hidden_state=last_decoder_states,
        )
