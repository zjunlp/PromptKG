from email.policy import default
from functools import partial
import os
import json
from dataclasses import dataclass
from random import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections import defaultdict
from enum import Enum
import time
import torch
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, BertTokenizer, T5Tokenizer, T5TokenizerFast
import numpy as np

from .processor import KGT5Dataset, KGCDataset
from .base_data_module import BaseDataModule

def lmap(f, x):
    return list(map(f, x))


class KGT5DataModule(BaseDataModule):
    def __init__(self, args, model) -> None:
        super().__init__(args)
        if "T5" in args.model_name_or_path:
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)

        self.entity2id = self.setup_tokenizer(self.tokenizer)
        self.entity2id = {int(k):v for k, v in self.entity2id.items()}


    def setup(self, stage=None):
        if stage == "fit":
            self.data_train = KGT5Dataset(self.args, self.tokenizer, mode="train")
            self.data_val = KGT5Dataset(self.args, self.tokenizer, mode="dev")
        else:
            self.data_test = KGT5Dataset(self.args, self.tokenizer, mode="test")
        
        self.filter_hr_to_ent = defaultdict(list)
        for mode in ["train", "dev", "test"]:
            with open(f"dataset/{self.args.dataset}/{mode}.tsv") as file:
                for line in file.readlines():
                    h, r, t = lmap(int,line.strip().split('\t'))
                    self.filter_hr_to_ent[(h,r)].append(t)
                    self.filter_hr_to_ent[(t,r)].append(h)
        
        for k in self.filter_hr_to_ent:
            self.filter_hr_to_ent[k] = list(set(self.filter_hr_to_ent[k]))
        
        entity2text = []
        with open(f"dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                line = line.strip().split("\t")[1]
                entity2text.append(line)
        
        self.entity_strings = entity2text
        self.tokenized_entities = self.tokenizer(entity2text, padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")

    def prepare_data(self):
        # use train.txt and entity2text to construct dataset, split by '\t'
        entity2text = []
        with open(f"dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                line = line.strip().split("\t")[1]
                entity2text.append(line)
        
        relation2text = []
        with open(f"dataset/{self.args.dataset}/relation2text.txt") as file:
            for line in file.readlines():
                line = line.strip().split("\t")[1]
                relation2text.append(line)
        
        
        def convert_triple_to_text(mode):
            train = []
            if self.args.overwrite_cache or not os.path.exists(f"dataset/{self.args.dataset}/{mode}.txt"):
                with open(f"dataset/{self.args.dataset}/{mode}.tsv") as file:
                    for line in file.readlines():
                        h, r, t = lmap(int,line.strip().split('\t'))
                        train.append("\t".join([entity2text[h] + " " + relation2text[r], " ".join(self.entity2id[t]), str(h), str(r), str(t)]))
                        train.append("\t".join([entity2text[t] + " " + relation2text[r] + " reversed", " ".join(self.entity2id[h]), str(t), str(r), str(h)]))
                
                with open(f"dataset/{self.args.dataset}/{mode}.txt", 'w') as file:
                    for l in train:
                        file.write(l+"\n")

        for m in ["train", "dev", "test"]:
            convert_triple_to_text(m)
        
        



    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)
        parser.add_argument("--chunk_size", type=int, default=128)
        return parser

    def get_tokenizer(self):
        return self.tokenizer

    def setup_tokenizer(self, tokenizer=None):
        #TODO 聚类
        #TODO add special token into vocab
        entity2id_path = os.path.join("dataset", f"{self.args.dataset}/entity2id.json")
        c = 10
        entity_list = [[f"[ENTITY_{j}{i}]" for i in range(c)] for j in range(10)]
        len_tokens = len(tokenizer)
        tmp_entity_list = []
        for t in entity_list:
            tmp_entity_list += t
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': tmp_entity_list})
        if not self.args.overwrite_cache and os.path.exists(entity2id_path):
            return json.load(open(entity2id_path))

        entity2id = defaultdict(list)
        model_path = "/newdisk1/xx/AI/kgc_nlu/output/FB15k-237/epoch=15-step=19299-Eval/hits10=0.97.ckpt"
        state_dict = torch.load(model_path, map_location="cpu")
        #! to be named in parameter
        entity_embedding = state_dict["state_dict"]["model.bert.embeddings.word_embeddings.weight"][30522:45473]
        entity_embedding = np.array(entity_embedding)


        # with open(f"entity_embedding/{self.args.dataset}.npy") as file:
        #     # [num_entity, hidden_size]
        #     entity_embedding = np.load(file)
        t = time.time()
        def generate_structure(entity_embedding, ids, cnt):
            kmeans = KMeans(n_clusters=c, random_state=self.args.seed).fit(entity_embedding)
            for i in range(c):
                labels = kmeans.labels_
                tmp_embedding = []
                tmp_ids = []
                now_id = 0
                for _,l in zip(ids, labels):
                    if l == i:
                        entity2id[_].append(entity_list[cnt][l])
                        tmp_embedding.append(entity_embedding[now_id])
                        tmp_ids.append(_)
                    now_id += 1
                tmp_embedding = np.array(tmp_embedding)
                if tmp_embedding.shape[0] > c:
                    generate_structure(tmp_embedding, tmp_ids, cnt+1)
                else:
                    for _, id_ in enumerate(tmp_ids):
                        entity2id[id_].append(entity_list[cnt][_])

        generate_structure(entity_embedding, ids=[_ for _ in range(len(entity_embedding))], cnt=0)
        print(f"cluster cost time : {time.time()-t}s")
        # for k in entity2id.keys():
        #     # change id to real virtual tokens
        #     entity2id[k] = [entity_list[_] for _ in entity2id[k]]
        with open(entity2id_path, "w") as file:
            json.dump(entity2id, file)

        return entity2id

    def collate_fn(self, items, mode):
        inputs = [item[0] for item in items]
        outputs = [item[1] for item in items]
        hr_pairs = [item[2] for item in items]
        target_entity_id = [item[3] for item in items]
        inputs_tokenized = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        outputs_tokenized = self.tokenizer(outputs, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        labels, labels_attention_mask = outputs_tokenized.input_ids, outputs_tokenized.attention_mask
        # for labels, set -100 for padding
        if mode == "train": labels[labels==self.tokenizer.pad_token_id] = -100
        # labels = -100 * torch.ones(labels.shape, dtype=torch.long)
        if mode == "train":
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, hr_pair=hr_pairs, target_entity_id=target_entity_id)
        
    

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
            collate_fn=partial(self.collate_fn, mode="train"), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="dev"), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="test"), pin_memory=True)

# def construct_mask(row_exs: List, col_exs: List = None, ) -> torch.tensor:
#     positive_on_diagonal = col_exs is None
#     num_row = len(row_exs)
#     col_exs = row_exs if col_exs is None else col_exs
#     num_col = len(col_exs)

#     # exact match
#     row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs])
#     col_entity_ids = row_entity_ids if positive_on_diagonal else \
#         torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])
#     # num_row x num_col
#     triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
#     if positive_on_diagonal:
#         triplet_mask.fill_diagonal_(True)

#     # mask out other possible neighbors
#     for i in range(num_row):
#         head_id, relation = row_exs[i].head_id, row_exs[i].relation
#         neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
#         # exact match is enough, no further check needed
#         if len(neighbor_ids) <= 1:
#             continue

#         for j in range(num_col):
#             if i == j and positive_on_diagonal:
#                 continue
#             tail_id = col_exs[j].tail_id
#             if tail_id in neighbor_ids:
#                 triplet_mask[i][j] = False

#     return triplet_mask


# def construct_self_negative_mask(exs: List) -> torch.tensor:
#     mask = torch.ones(len(exs))
#     for idx, ex in enumerate(exs):
#         head_id, relation = ex.head_id, ex.relation
#         neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
#         if head_id in neighbor_ids:
#             mask[idx] = 0
#     return mask.bool()


class SimKGCDataModule(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)

    
    def setup(self, stage=None):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = KGCDataset(self.args, mode="train")
            self.data_val = KGCDataset(self.args, mode="dev")
        else:
            self.data_test = KGCDataset(self.args, mode="test")
        
        self.filter_hr_to_t = defaultdict(list)
        self.filter_tr_to_h = defaultdict(list)


        for mode in ["train", "dev", "test"]:
            with open(f"dataset/{self.args.dataset}/{mode}.tsv") as file:
                for line in file.readlines():
                    h, r, t = lmap(int,line.strip().split('\t'))
                    self.filter_hr_to_t[(h,r)].append(t)
                    self.filter_tr_to_h[(t,r)].append(h)
        
        self.filter_hr_to_t = {k: list(set(v)) for k, v in self.filter_hr_to_t.items()}
        self.filter_tr_to_h = {k: list(set(v)) for k, v in self.filter_tr_to_h.items()}
        max_filter_ent = max(max([len(_) for _ in self.filter_hr_to_t.values()]), max([len(_) for _ in self.filter_tr_to_h.values()]))
        print("=== max filter ent {} ===".format(max_filter_ent))
        
        entity2text = []
        with open(f"dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                line = line.strip().split("\t")[1]
                entity2text.append(line)
        
        self.entity_strings = entity2text
        # self.tokenized_entities = self.tokenizer(entity2text, padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")

        self.entity2text = self.get_entity_to_text()
        self.relation2text = self.get_relation_to_text()
        print("finished data processing... costing {}s...".format(time.time() - now_time))


    def prepare_data(self):
        # use train.txt and entity2text to construct dataset, split by '\t'
        pass

    def get_entity_dataloader(self):
        num_entity = len(self.entity2text)
        self.entity_dataset = TensorDataset(torch.arange(num_entity))

        return DataLoader(self.entity_dataset, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, 
            collate_fn=self.test_collate_fn, pin_memory=True)
    
    def get_entity_to_text(self):
        entity2text = {}
        with open(f"./dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                entity2text[int(id_)] = text
        return entity2text
    
    def get_relation_to_text(self):
        relation2text = {}
        with open(f"./dataset/{self.args.dataset}/relation2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                relation2text[int(id_)] = text
        
        return relation2text

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)
        return parser
    
    def test_collate_fn(self, items):
        # items = items.tolist()
        inputs = self.tokenizer([self.entity2text[_[0].item()] for _ in items],padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")
        return inputs
    
    def collate_fn(self, items, mode):

        def convert_triple_to_text(triple):
            h, r = triple.hr
            t = triple.t
            inverse = triple.inverse
            r_input = self.relation2text[r]

            if inverse:
                r_input = " inverse " + r_input

            h_input = self.entity2text[h]
            t_input = self.entity2text[t]

            return h_input, r_input, t_input
        

        text = [convert_triple_to_text(_) for _ in items]
        h_inputs = [_[0] for _ in text]
        r_inputs = [_[1] for _ in text]
        t_inputs = [_[2] for _ in text]


        hr_inputs = self.tokenizer(h_inputs, r_inputs, padding='max_length', truncation="longest_first", max_length=2*self.args.max_entity_length, return_tensors="pt")
        t_inputs = self.tokenizer(t_inputs, padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")
        h_inputs = self.tokenizer(h_inputs, padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")



        return {'hr_token_ids': hr_inputs['input_ids'],
            'hr_token_type_ids': hr_inputs['token_type_ids'],
            'hr_mask': hr_inputs['attention_mask'],
            'tail_token_ids': t_inputs['input_ids'],
            'tail_token_type_ids': t_inputs['token_type_ids'],
            'tail_mask': t_inputs.attention_mask,
            'head_token_ids': h_inputs['input_ids'],
            'head_token_type_ids': h_inputs['token_type_ids'],
            'head_mask': h_inputs.attention_mask,
            'triplet_mask': None, #TODO rerank op
            "self_negative_mask": None,
            "batch_data": items
        }
                #  'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
                # 'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
                #TODO I don't know how to do!
                # 'obj': items}
        
    
    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
            collate_fn=partial(self.collate_fn, mode="train"), pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="dev"), pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="test"), pin_memory=True, drop_last=False)
