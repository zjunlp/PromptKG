from logging import debug
import os
import copy
import random
import pytorch_lightning as pl
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from transformers.utils.dummy_pt_objects import PrefixConstrainedLogitsProcessor

from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from functools import partial
from .utils import rank_score, acc

from models.trie import get_end_to_end_prefix_allowed_tokens_fn_hf

from models import Trie
from models.model import BartKGC
from models.trie import get_trie
from models.utils import get_entity_spans_pre_processing

from typing import Callable, Iterable, List

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    def __init__(self, model:BartKGC, args, tokenizer=None):
        super().__init__(model, args)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters(args)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.best_acc = 0
        self.first = True
        
        # load trie for each dataset
        # with open(f"{args.data_dir}/trie.pkl", "rb") as file:
        #     self.trie = pickle.load(file)
        self.tokenizer = tokenizer
        self._get_entities()
        self.entity_trie = get_trie(args, tokenizer=tokenizer)


        self.last_filter_ent = []



        self.mention_trie = Trie([tokenizer(f" <entity>").input_ids[1:]])

        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.decode = partial(decode, tokenizer=self.tokenizer)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        batch.pop("filter_ent_ids")
        bsz = batch['input_ids'].shape[0]
        loss = self.model(**batch, use_cache=False, return_dict=True).loss
        self.log("Train/loss", loss)
        return loss
        
    def _get_ranks(self, outputs, labels):
        bsz = len(labels)
        
        tmp = []
        outputs = [self.decode(_) for _ in outputs]
        # for o in outputs:
        #     t = self.decode(output_ids = o)
        #     tmp.append(t)
        # outputs = tmp
        labels = self.decode(output_ids = labels)
        # labels = [_
        # tmp = []
        # for o in ent:
        #     t = self.decode(output_ids = o)
        #     tmp.append(t)
        # ent = tmp
        ranks = []
        for i in range(bsz):
            # get the real entity
            label = labels[i]
            output = outputs[i]
            # filter entities to ignore
            has_flag = False
            real_idx = 1
            for j in range(len(output)):
                if label in output[j]:
                    has_flag = True
                    ranks.append(real_idx)
                    break
                real_idx += 1
            #TODO more youmei solution
            if not has_flag: ranks.append(10000)
        return ranks
    
    def _eval(self, batch, batch_idx, ):
        labels = batch.pop("labels").cpu()
        decoder_input_ids = batch.pop("decoder_input_ids")
        # ent id for filter
        ent = batch.pop("filter_ent_ids")
        if "input_sentences" in batch: input_sentences = batch.pop("input_sentences") 
        topk = self.args.beam_size
        bsz = batch['input_ids'].shape[0]
        # clean filter
        import time
        for i in range(bsz):
            now_label = [self.tokenizer.eos_token_id] + labels[i].tolist()
            for e in self.last_filter_ent:
                self.entity_trie.add(self.id2ent[e])
           
            # delete token
            self.last_filter_ent = []
            ent[i] = list(set(ent[i]))
            for e in ent[i]:
                text_ids = self.id2ent[e]
                if text_ids == now_label: continue
                cnt_ids = 0
                now = self.entity_trie.trie_dict
                t_now = now
                pop_ids = -1
                while cnt_ids < len(text_ids) - 2:
                    if text_ids[cnt_ids] not in now: break
                    now = now[text_ids[cnt_ids]]
                    if len(now.keys()) > 1:
                        t_now = now
                        pop_ids = text_ids[cnt_ids+1]
                    cnt_ids += 1
                try:
                    t_now.pop(pop_ids)
                except:
                    pass
        self.last_filter_ent = ent[0]
        
        prefix_allowed_tokens_fn = None
        # if self.args.prefix_tree_decode and not self.args.output_full_sentence:
        prefix_allowed_tokens_fn = lambda batch_id, sent: self.entity_trie.get(sent.tolist())
        # elif self.args.prefix_tree_decode:
        #     prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(
        #         tokenizer=self.tokenizer,
        #         sentences=get_entity_spans_pre_processing(input_sentences),
        #         mention_trie=self.mention_trie,
        #         candidates_trie=self.entity_trie, 
        #     )
        # assert  not (self.args.prefix_tree_decode ^ (prefix_allowed_tokens_fn is None)), "use prefix tree decode must determine fn"

        
        outputs = self.model.generate(
            **batch,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            max_length=64,
            forced_bos_token_id=self.args.bos_token_id,
            decoder_start_token_id=self.args.bos_token_id,
            use_cache=True,
        ).view(bsz, topk, -1).cpu()
    
        # ranks = self._get_ranks(outputs, labels, ent)
        if batch_idx % 100 == 0:
            generated_text = ["\n".join(self.decode(output_ids=_[:5])) for _ in outputs]
            outputs_ = self.decode(output_ids=self.model.generate(**batch, forced_bos_token_id=self.args.bos_token_id, use_cache=True, max_length=40))
            # self.log("conditonal output",generated_text)
            input_text = self.decode(output_ids=batch['input_ids'])
            label_text = self.decode(output_ids=decoder_input_ids)
            for i in range(bsz):
                print(f"origin input : \t{input_text[i]}")
                print(f"model output : \t{outputs_[i]}")
                print(f"model constrined output : \t{generated_text[i]}")
                print(f"true label : \t{label_text[i]}")
                print("*"*20)

        return dict(outputs=outputs, labels=labels.cpu())

    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        # self.log("Eval/loss", np.mean(ranks))
        return result

    def validation_epoch_end(self, outputs) -> None:
            
        labels = [_ for o in outputs for _ in o['labels']]
        outputs = [_ for o in outputs for _ in o['outputs']]
        ranks = self._get_ranks(outputs, labels)
        ranks = np.array(ranks)
        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval/hits10", hits10, prog_bar=True, on_epoch=True)
        self.log("Eval/hits1", hits1)
        self.log("Eval/hits3", hits3)
        self.log("Eval/hits20", hits20)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # ranks = self._eval(batch, batch_idx)
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))

        return result
        # return {"test_rank": np.array(ranks)}

    def test_epoch_end(self, outputs) -> None:
        # ranks = np.concatenate([o["test_rank"] for o in outputs]).reshape(-1)

        labels = [_ for o in outputs for _ in o['labels']]
        outputs = [_ for o in outputs for _ in o['outputs']]
        ranks = self._get_ranks(outputs, labels)
        ranks = np.array(ranks)

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

       
        self.log("Test/hits10", hits10)
        self.log("Test/hits20", hits20)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
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

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--output_full_sentence", type=int, default=0, help="decode full sentence not entity")
        parser.add_argument("--entity_token", type=str, default="<entity>", help="")
        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--prefix_tree_decode", type=int, default=1, help="")
        parser.add_argument("--beam_size", type=int, default=30, help="")
        
        return parser
    

    def _get_entities(self):
        self.id2ent = []
        with open(os.path.join(self.args.data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            cnt = 0
            for line in ent_lines:
                temp = line.strip().split('\t')
                self.id2ent.append(temp[1])
        
        for i, e in enumerate(self.id2ent):
            t = [self.tokenizer.eos_token_id] + self.tokenizer(e).input_ids
            self.id2ent[i] = t
