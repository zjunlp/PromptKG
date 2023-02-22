from logging import debug
from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

import torch
import torch.nn as nn

import torch.nn.functional as F

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertForSequenceClassification




class T5KGC(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()

   
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        
        return parser
    
class T5KBQAModel(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()

   
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        
        return parser

class Sparsemax(nn.Module):
    """Sparsemax loss"""

    def __init__(self, k_sparse=1):
        super(Sparsemax, self).__init__()
        self.k_sparse = k_sparse
        
    def forward(self, preds, labels):
        """
        Args:
            preds (torch.Tensor):  [batch_size, number_of_logits]
            labels (torch.Tensor): [batch_size] index, not ont-hot
        Returns:
            torch.Tensor
        """
        preds = preds.reshape(preds.size(0), -1) # [batch_size, -1]
        topk = preds.topk(self.k_sparse, dim=1)[0] # [batch_size, k_sparse]
        
        # log(sum(exp(topk)))
        pos_loss = torch.logsumexp(topk, dim=1)
        # s_t
        neg_loss = torch.gather(preds, 1, labels[:, None].expand(-1, preds.size(1)))[:, 0]
        
        return (pos_loss - neg_loss).mean()

class KGBERTModel(BertForSequenceClassification):
    @staticmethod
    def add_to_argparse(parser):
        return parser
class BartKGC(BartForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.loss_fn = Sparsemax(10)
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_label_type", type=int, default=0, help="")
        parser.add_argument("--pretrain", type=int, default=0, help="")
        
        return parser

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            mask = labels.unsqueeze(2).expand(lm_logits.shape)
            lm_logits = torch.masked_select(lm_logits, mask !=-100)
            labels = torch.masked_select(labels, labels != -100)
            masked_lm_loss = self.loss_fn(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class KGRECModel(BertForMaskedLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser