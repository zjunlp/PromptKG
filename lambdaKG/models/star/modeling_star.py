import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class BertForPairCls(AutoModelForSequenceClassification):
    def __init__(self, config):
        super(BertForPairCls, self).__init__(config)
        self.num_labels = self.config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size * 4, self.num_labels)  # 4个 rep_src, rep_tgt, src - tgt, src * tgt
        if hasattr(self, "roberta"):
            self.bert = self.roberta

    def encoder(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output

    def classifier(self, rep_src, rep_tgt):
        cls_feature = torch.cat(
            [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1
        )
        cls_feature = self.dropout(cls_feature)
        logits = self.proj(cls_feature)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, src_input_ids, tgt_input_ids,
                src_attention_mask=None, tgt_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None,
                labels=None, *args, **kwargs):
        rep_src = self.encoder(src_input_ids, attention_mask=src_attention_mask, token_type_ids=src_token_type_ids)
        rep_tgt = self.encoder(tgt_input_ids, attention_mask=tgt_attention_mask, token_type_ids=tgt_token_type_ids)
        logits = self.classifier(rep_src, rep_tgt)
        outputs = (logits,)

        if labels is not None:
            if hasattr(self.config, "pos_gain"):
                loss_weight = get_pos_gain(
                    self.num_labels, labels, pos_weight=getattr(self.config, "pos_gain"), neg_weight=1.,
                    float_dtype=logits.dtype)
            else:
                loss_weight = None

            if self.num_labels == 1:
                #  We are doing sigmoid instead of regression
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
                loss = loss_fct(logits.view(-1), labels.view(-1).to(logits.dtype))
            else:
                loss_fct = nn.CrossEntropyLoss(weight=loss_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return 