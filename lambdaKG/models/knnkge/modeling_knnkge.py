from transformers import BertForMaskedLM
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from .large_embedding import LargeEmbedding

class KNNKGEModel(BertForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser




class KNNKGEModel_MIX(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # add gpu-cpu large embedding
        emb_name = getattr(config, "emb_name", 'entity_emb')
        self.ent_embeddings = LargeEmbedding(ip_config='emb_ip.cfg', emb_name='entity_emb', lr=config.lr * 10, num_emb=config.num_ent)

        # tie the weight to ent_lm_head
        self.ent_lm_head = EntLMHead(config)
        self.ent_lm_head.transform = self.cls.predictions.transform

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        parser.add_argument("--num_ent", type=int, default=60000, help="")
        return parser
    

    def convert_ent_embeddings_to_embeddings(self):
        self.ent_embeddings = self.ent_embeddings.to_embeddings()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ent_pos : torch.Tensor = None, # (batch_size, 1)
        mask_pos : torch.Tensor = None,
        ent_masked_lm_labels=None, 
        ent_index=None,
        is_test=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        bsz = input_ids.shape[0]
        ent_ids = input_ids[torch.arange(bsz), ent_pos]
        ent_embeddings = self.ent_embeddings(ent_ids.unsqueeze(1)).squeeze(1)
        # pad token for placeholder
        input_ids[torch.arange(bsz), ent_pos] = 1
        word_embeddings = self.bert.embeddings(input_ids)

        

        # replace the ent embedding
        for i in range(bsz):
            # word_embeddings[i][ent_pos[i]] = ent_embeddings[i].clone().detach().requires_grad_(True)
            word_embeddings[i][ent_pos[i]] = ent_embeddings[i]

        inputs_embeds = word_embeddings




        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # prediction_scores = self.cls(sequence_output)

        if not is_test:
            ent_cls_weight = self.ent_embeddings(ent_index.view(1,-1)).squeeze()
        else:
            ent_cls_weight = self.ent_embeddings.weight
        ent_logits = self.ent_lm_head(sequence_output[torch.arange(bsz), mask_pos] ,
                                      ent_cls_weight)
        # TODO: add full daataset for eval 
        _, outputs = torch.sort(ent_logits, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), ent_masked_lm_labels].detach() + 1

        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        ent_masked_lm_loss = loss_fct(ent_logits.view(-1, ent_logits.size(-1)), ent_masked_lm_labels.view(-1))
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()  # -100 index = padding token
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # if not return_dict:
        #     output = (prediction_scores,) + outputs[2:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return dict(
            loss=ent_masked_lm_loss,
            ranks = ranks
        )


class EntLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.transform = None
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, weight, **kwargs):
        # x = self.dense(features)
        # x = self.gelu(x)
        # # x = self.dropout(x)
        # x = self.layer_norm(x)
        x = self.transform(features)
        x = x.matmul(weight.t())

        return x