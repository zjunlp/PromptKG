from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from IPython import embed

from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup

from functools import partial
from .utils import rank_score, acc



def multilabel_categorical_crossentropy(y_pred, y_true):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        # self.num_training_steps = data_config["num_training_steps"]
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.best_acc = 0
        self.first = True


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask , token_type_ids, labels = batch
        # embed();exit()
        logits = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def ___validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, entity_idx, ref_entity_idx, modified_entity_ids, modified_attention_mask = batch.values()
        entity_idx = entity_idx.view(-1)
        ref_entity_idx = ref_entity_idx.view(-1)

        # get the label score, firstly
        label_score = self.model.predict(input_ids.squeeze(0), attention_mask, entity_idx=entity_idx, ref_entity_idx=ref_entity_idx)[0]
        # the number of candidates
        len_sample = modified_entity_ids.shape[1]
        template_input_ids = input_ids.squeeze(0).clone()
        template_attention_mask = attention_mask.squeeze(0).clone()
        # input_ids = input_ids.squeeze(0).repeat((len_sample,1))
        # attention_mask = attention_mask.squeeze(0).repeat((len_sample,1))
        modified_entity_ids = modified_entity_ids.squeeze(0)
        modified_attention_mask = modified_attention_mask.squeeze(0)

        def _copy_everything(a, b, c, d, s):
            a = a.repeat(s, 1)
            b = b.repeat(s, 1)
            a[:, entity_idx:entity_idx+num_tokens] = c[:, 1:]
            b[:, entity_idx:entity_idx+num_tokens] = d[:, 1:]

            return a, b



        num_tokens = 17


        # for i in range(len_sample):
        #     input_ids[i][entity_idx:entity_idx+num_tokens] = modified_entity_ids[i][1:]
        #     attention_mask[i][entity_idx:entity_idx+num_tokens] = modified_attention_mask[i][1:]

        # input_ids[:, entity_idx:entity_idx+num_tokens] = modified_entity_ids[:, 1:]
        # attention_mask[:, entity_idx:entity_idx+num_tokens] = modified_attention_mask[:, 1:]
        all_rank = []

        bb_size = 64


        for i in range(0, int(len_sample/bb_size)+1):
            st = i*bb_size
            ed = min((i+1)*bb_size, len_sample)
            a, b = _copy_everything(template_input_ids, template_attention_mask, modified_entity_ids[st:ed], modified_attention_mask[st:ed], ed-st)
            ranks = self.model.predict(input_ids=a, attention_mask=b, entity_idx=entity_idx, ref_entity_idx=ref_entity_idx)
            all_rank.append(ranks.squeeze(1))
        _, indices = torch.sort(torch.cat([label_score, torch.cat(all_rank, dim=0)], dim=0), descending=True)
        rank = (indices[0]+1).unsqueeze(0)


        # _, indices = torch.sort(torch.cat([label_rank.unsqueeze(1).detach().cpu(), all_rank.detach().cpu()], dim=1))

        return {"eval_rank": rank.detach().cpu().numpy()}


        
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask , token_type_ids, labels = batch
        # embed();exit()
        logits = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)

        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}

    
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        result = acc(logits, labels)
        if self.first:
            result = 0
            self.first = False
        
        self.best_acc = max(self.best_acc, result)
        
        self.log("Eval/acc", result, prog_bar=True, on_epoch=True)
        self.log("Eval/best_acc", self.best_acc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, candidates_ids = batch
        candidates_ids = candidates_ids[0]
        label_rank, all_rank = self.model.predict(input_ids, attention_mask, labels=labels, candidates_ids=candidates_ids)

        _, indices = torch.sort(torch.cat([label_rank.unsqueeze(1).detach().cpu(), all_rank.detach().cpu()], dim=1))
        rank = indices[:,0] + 1
        
        return {"test_rank": rank.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([o["test_rank"] for o in outputs])
       
        hits10, hits5, hits1, mrr = rank_score(ranks)
        self.log("Test/hits10", hits10)
        self.log("Test/hits5", hits5)
        self.log("Test/hits1", hits1)
        self.log("Test/mrr", mrr)

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if "Get" not in self.args.model_class:
            optimizer_group_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
            ]
        else:
            optimizer_group_parameters = [
                {"params": [p for n, p in self.model.classifier.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in self.model.classifier.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
            ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
