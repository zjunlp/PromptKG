import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaLMHead, RobertaModel

import logging
logger = logging.getLogger(__name__)


class RobertaForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)

        self.virtual_demo = config.virtual_demo
        self.virtual_demo_length_per_label = config.virtual_demo_length_per_label
        self.virtual_demo_init = config.virtual_demo_init
        self.total_label_tokens = config.total_label_tokens
        
        self.virtual_demo_embed = None
        if config.virtual_demo:
            self.virtual_demo_embed = nn.Embedding(self.virtual_demo_length_per_label * config.num_labels, config.hidden_size)

        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.tokenizer = None

        # For regression
        self.lb = None
        self.ub = None

    def embed_encode(self, input_ids):
        embedding_output = self.roberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def _init_prompt(self):
        if self.virtual_demo_embed is not None:
            if self.virtual_demo_init == 'random':
                self.virtual_demo_embed.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif self.virtual_demo_init == 'vocab':
                rand_id = torch.randint(100, self.config.vocab_size, (self.num_labels * self.virtual_demo_length_per_label)).long()
                rand_emb = self.embed_encode(rand_id)
                self.virtual_demo_embed = self.virtual_demo_embed.from_pretrained(rand_emb, freeze=False)
            else:
                raise ValueError("invalid initilization method.")

    def generate_virtual_demo_inputs(
        self, 
        input_ids, 
        block_flag_for_demo
    ):
        input_embeds = self.embed_encode(input_ids)

        if self.virtual_demo:
            virtual_demo_indices = block_flag_for_demo[block_flag_for_demo > 0] - 1
            replaced_embeds = self.virtual_demo_embed(virtual_demo_indices)
            input_embeds[block_flag_for_demo > 0] = replaced_embeds
    
        return input_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        block_flag_for_demo=None,
        labels=None,
        input_embeds=None,
        return_output=False,
        reduction='mean',
        only_mask_output=False
    ):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        if self.virtual_demo:
            input_embeds = self.generate_virtual_demo_inputs(input_ids, block_flag_for_demo)
        
            outputs = self.roberta(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds
            )
        else:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        sequence_output, _ = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # early exit when calculate contrastive representation
        if only_mask_output:
            outputs = (sequence_mask_output)
            return outputs

        loss = None
        prediction_mask_scores = self.lm_head(sequence_mask_output)
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_output:
            outputs = (logits, sequence_mask_output)
        else:
            outputs = (logits, )
    
        return ((loss,) + outputs) if loss is not None else outputs