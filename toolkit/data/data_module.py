from functools import partial
import os
import json
from dataclasses import dataclass
from random import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections import defaultdict
from enum import Enum
import time
import random
from sklearn import neighbors
import torch
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, BertTokenizer, T5Tokenizer, T5TokenizerFast
import numpy as np

from models.utils import construct_mask

from .processor import KGT5Dataset, KGCDataset, PretrainKGCDataset, LAMADataset, LAMASampler
from .base_data_module import BaseKGCDataModule, QADataModule, BaseKGRECDataModule
from .rec_processor import KGRECDataset, PretrainKGRECDataset
from .utils import LinkGraph, Roberta_utils

def lmap(f, x):
    return list(map(f, x))


class KGT5DataModule(BaseKGCDataModule):
    def __init__(self, args, lama: bool=False) -> None:
        super().__init__(args, lama)
        if "t5" in args.model_name_or_path.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
            
        # use entity plain name as labels, "Plato, a xxx" => "Plato"
        self.entity2input_ids = {i:k for i, k in enumerate(self.tokenizer(list(map(lambda x: x.split(",")[0], self.entity2text.values())), add_special_tokens=True).input_ids)}

        decoder_start_token_id = self.tokenizer.eos_token_id if "bart" in self.args.model_name_or_path else self.tokenizer.pad_token_id
        self.entity2input_ids = {i: [decoder_start_token_id] + k for i, k in self.entity2input_ids.items()}



    @staticmethod
    def add_to_argparse(parser):
        BaseKGCDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        return parser

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
        def convert_triple_to_text(triple):
            h, r = triple.hr
            t = triple.t
            inverse = triple.inverse
            r_input = self.relation2text[r]

            if inverse:
                r_input = " [inverse] " + r_input

            h_input = self.entity2text[h]
            # ! warning, hard coded
            t_input = self.entity2text[t].split(",")[0]

            return h_input, r_input, t_input
        

        text = [convert_triple_to_text(_) for _ in items]
        inputs_h = [_[0] for _ in text]
        inputs_r = [_[1] for _ in text]
        outputs = [_[2] for _ in text]

        inputs_tokenized = self.tokenizer(inputs_h, inputs_r, padding='max_length', truncation="longest_first", max_length=self.args.max_seq_length, return_tensors="pt")
        outputs_tokenized = self.tokenizer(outputs, padding='max_length', truncation=True, max_length=self.args.max_seq_length, return_tensors="pt")
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        labels, labels_attention_mask = outputs_tokenized.input_ids, outputs_tokenized.attention_mask
        # for labels, set -100 for padding
        if mode == "train": labels[labels==self.tokenizer.pad_token_id] = -100
        # labels = -100 * torch.ones(labels.shape, dtype=torch.long)
        if mode == "train":
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, batch_data=items)
        
    

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
            collate_fn=partial(self.collate_fn, mode="train"), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="dev"), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="test"), pin_memory=True)


class MetaQADataModule(QADataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True, sep_token="<sep>")
        self.tokenizer.add_tokens(["[inverse]"], special_tokens=True)
    
    @staticmethod
    def add_to_argparse(parser):
        BaseKGCDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)
        parser.add_argument("--k_hop", type=int, default=1)
        return parser

    def collate_fn(self, items, mode):

        question = [_['question'] for _ in items]
        # h sep r sep t
        answer = [self.tokenizer.sep_token.join(_['triples'][0].values()) for _ in items]
        inputs = self.tokenizer(question, padding='longest', 
            truncation="longest_first", max_length=self.args.max_seq_length, return_tensors="pt").input_ids
        labels = self.tokenizer(answer, padding='longest', 
            truncation="longest_first", max_length=self.args.max_seq_length, return_tensors="pt").input_ids

        
        return dict(input_ids=inputs, labels=labels)


        
class KGBERTDataModule(BaseKGCDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)
    
    def collate_fn(self, items, mode):
        def convert_triple_to_text_positive(triple):
            h, r = triple.hr
            t = triple.t
            inverse = triple.inverse

            r_input = self.relation2text[r]
            h_input = self.entity2text[h]
            t_input = self.entity2text[t]

            # ==== construct the negative samples
            if inverse:
                h, t = t, h
            else:
                pass
                

            return self.tokenizer.sep_token.join([h_input, r_input, t_input]), 1
        # negative sampling here
        def convert_triple_to_text_negative(triple):
            h, r = triple.hr
            t = triple.t
            inverse = triple.inverse

            r_input = self.relation2text[r]
            h_input = self.entity2text[h]
            t_input = self.entity2text[t]

            # ==== construct the negative samples
            if inverse:
                h, t = t, h
                while True:
                    h = random.randint(0, self.num_entity-1)
                    if h not in self.filter_tr_to_h[triple.hr]: break
                h_input = self.entity2text[h]
                # corrupt head
            else:
                while True:
                    t = random.randint(0, self.num_entity-1)
                    if t not in self.filter_hr_to_t[triple.hr]: break
                t_input = self.entity2text[t]


            return self.tokenizer.sep_token.join([h_input, r_input, t_input]), 0



        text_pos = [convert_triple_to_text_positive(_) for _ in items]
        text_neg = [convert_triple_to_text_negative(_) for _ in items]
        inputs = [_[0] for _ in text_pos] + [_[0] for _ in text_neg]
        labels = [_[1] for _ in text_pos] + [_[1] for _ in text_neg]

        inputs = self.tokenizer(inputs, padding='longest', truncation=True, max_length=self.args.max_seq_length, return_tensors="pt")

        inputs.update(labels=torch.tensor(labels, dtype=torch.long))
        return dict(inputs)
    
    @staticmethod
    def add_to_argparse(parser):
        BaseKGCDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        return parser









class SimKGCDataModule(BaseKGCDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)

    
    


    def prepare_data(self):
        # use train.txt and entity2text to construct dataset, split by '\t'
        pass

    def get_entity_dataloader(self):
        num_entity = len(self.entity2text)
        self.entity_dataset = TensorDataset(torch.arange(num_entity))

        return DataLoader(self.entity_dataset, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, 
            collate_fn=self.test_collate_fn, pin_memory=True)
    
    

    @staticmethod
    def add_to_argparse(parser):
        BaseKGCDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)
        return parser
    
    def test_collate_fn(self, items):
        # items = items.tolist()
        inputs = self.tokenizer.batch_encode_plus([self.entity2text[_[0].item()] for _ in items],padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")
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


        hr_inputs = self.tokenizer(h_inputs, r_inputs, padding='longest', truncation="longest_first", max_length=2*self.args.max_entity_length, return_tensors="pt")
        t_inputs = self.tokenizer(t_inputs, padding='longest', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")
        h_inputs = self.tokenizer(h_inputs, padding='longest', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")



        return {'hr_token_ids': hr_inputs['input_ids'],
            'hr_token_type_ids': hr_inputs['token_type_ids'],
            'hr_mask': hr_inputs['attention_mask'],
            'tail_token_ids': t_inputs['input_ids'],
            'tail_token_type_ids': t_inputs['token_type_ids'],
            'tail_mask': t_inputs.attention_mask,
            'head_token_ids': h_inputs['input_ids'],
            'head_token_type_ids': h_inputs['token_type_ids'],
            'head_mask': h_inputs.attention_mask,
            'triplet_mask': construct_mask(row_exs=items) if mode in ["train", "dev"] else None, #TODO rerank op
            "self_negative_mask": None,
            "batch_data": items
        }
                #  'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
                # 'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
                #TODO I don't know how to do!
                # 'obj': items}
        
    
    

class KNNKGEDataModule(BaseKGCDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)

        entity_list = [f"[entity{i}]" for i in range(self.num_entity)]
        relation_list = [f"[relation{i}" for i in range(self.num_relation)]
        self.st_entity = self.tokenizer.vocab_size
        self.ed_entity = self.tokenizer.vocab_size + self.num_entity
        self.st_realtion = self.tokenizer.vocab_size + self.num_entity
        self.ed_realtion = self.tokenizer.vocab_size + self.num_entity + self.num_relation
    
    @staticmethod
    def add_to_argparse(parser):
        BaseKGCDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)
        return parser
    
    def collate_fn(self, items, mode):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label = []
        filter_entity_ids = []
        for item_idx,item in enumerate(items):
            inverse = item.inverse
            h, r = item.hr
            t = item.t
            if not inverse:
                input_ = self.tokenizer(self.tokenizer.sep_token.join([self.tokenizer.pad_token, self.entity2text[h]]), 
                        self.tokenizer.sep_token.join([self.tokenizer.pad_token, self.relation2text[r], self.tokenizer.mask_token]),
                    padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
                        )
                cnt = 0
                for i in range(len(input_.input_ids)):
                    if input_.input_ids[i] == self.tokenizer.pad_token_id:
                        if cnt == 2:
                            break
                        if cnt == 1:
                            cnt += 1
                            input_.input_ids[i] = len(self.tokenizer) + self.num_entity + r
                        if cnt == 0:
                            cnt += 1
                            input_.input_ids[i] = len(self.tokenizer) + h
                filter_entity_ids.append(self.filter_hr_to_t[(h, r)])
                    



                input_ids.append(input_.input_ids)
                attention_mask.append(input_.attention_mask)
                token_type_ids.append(input_.token_type_ids)
            else:
                input_ = self.tokenizer(self.tokenizer.sep_token.join([self.tokenizer.mask_token, self.tokenizer.pad_token, self.relation2text[r]]), 
                        self.tokenizer.sep_token.join([self.tokenizer.pad_token, self.entity2text[h]]),
                    padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
                        )
                cnt = 0
                for i in range(len(input_.input_ids)):
                    if input_.input_ids[i] == self.tokenizer.pad_token_id:
                        if cnt == 2:
                            break
                        if cnt == 1:
                            cnt += 1
                            input_.input_ids[i] = h+ len(self.tokenizer)
                        if cnt == 0:
                            cnt += 1
                            input_.input_ids[i] = len(self.tokenizer) + self.num_entity + r
                input_ids.append(input_.input_ids)
                attention_mask.append(input_.attention_mask)
                token_type_ids.append(input_.token_type_ids)
                filter_entity_ids.append(self.filter_tr_to_h[(h, r)])
            
            filter_entity_ids[item_idx].remove(t)
            label.append(t)
        
        features =  dict(input_ids=input_ids, attention_mask=attention_mask, 
                token_type_ids=token_type_ids)
        
        features = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=self.args.max_seq_length,
            return_tensors="pt"
        )
        features.update(dict(label=torch.tensor(label)))
        if mode != "train":
            features.update(dict(filter_entity_ids=filter_entity_ids))
        return features




class KNNKGEPretrainDataModule(KNNKGEDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def setup(self, stage=None):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = PretrainKGCDataset(self.args, mode="train")
            self.data_val = PretrainKGCDataset(self.args, mode="dev")
        else:
            self.data_test = PretrainKGCDataset(self.args, mode="test")
        
        self.filter_hr_to_t = defaultdict(list)
        self.filter_tr_to_h = defaultdict(list)
        triple_train = KGCDataset(self.args, mode="train")
        self.graph = LinkGraph(triple_train)


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

        print("finished data processing... costing {}s...".format(time.time() - now_time))
    
    def collate_fn(self, items, mode):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label = []
        filter_entity_ids = []
        for item_idx,item in enumerate(items):
            e = item
            # retrieved neighbors from the knowledge graph
            neighbor_ids = list(self.graph.get_neighbor_ids(5))
            num_neighbor = len(neighbor_ids)
            words = self.entity2text[e].split()
            # st = random.randint(0, max(1,len(words)-40))
            # input_ = self.tokenizer(f"The description of {self.tokenizer.mask_token} is {words[st: min(st+200, len(words))]}",
            #     padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
            #         )
            # input_ids.append(input_.input_ids)
            # attention_mask.append(input_.attention_mask)
            # token_type_ids.append(input_.token_type_ids)
            # label.append(e)

            input_ = self.tokenizer(self.tokenizer.pad_token * num_neighbor + f"The description of {self.tokenizer.mask_token} is {words}.",
                padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
                    )
            # add offset for entity 
            for i in range(num_neighbor):
                # [CLS] [PAD] [PAD]....
                input_.input_ids[i+1] = neighbor_ids[i] + self.st_entity
            input_ids.append(input_.input_ids)
            attention_mask.append(input_.attention_mask)
            token_type_ids.append(input_.token_type_ids)
            label.append(e)

            # input_ = self.tokenizer(f"The description of is {words[st: min(st+200, len(words))]} ",f"{self.tokenizer.mask_token} ",
            #     padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
            #         )
            # input_ids.append(input_.input_ids)
            # attention_mask.append(input_.attention_mask)
            # token_type_ids.append(input_.token_type_ids)
            
            # label.append(e)
        
        features =  dict(input_ids=input_ids, attention_mask=attention_mask, 
                token_type_ids=token_type_ids)
        
        features = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=self.args.max_seq_length,
            return_tensors="pt"
        )
        features.update(dict(label=torch.tensor(label)))
        return features

class LAMADataModule(BaseKGCDataModule):
    def __init__(self, args, tokenizer=None, lama=True) -> None:
        super().__init__(args, lama)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False) \
                        if tokenizer == None else tokenizer
        if args.pelt:
            self.pelt = Roberta_utils(self.tokenizer)
            assert args.lamadataset in ['Google_RE', 'TREx'], 'Not support for datasets except Google_RE and TREx'
    
    @staticmethod
    def add_to_argparse(parser):
        BaseKGCDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)
        parser.add_argument("--lamadataset", type=str, default=None, choices=['Google_RE', 'Squad', 'TREx', 'ConceptNet', None], 
            help="Choose a subdataset in [Google_RE, Squad, TREx, ConceptNet] of LAMA or None (represents the full dataset)."
        )
        parser.add_argument("--pelt", type=int, default=0)
        return parser

    def setup(self, stage):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit" and False:
            self.data_train = LAMADataset(self.args)
            self.data_val = LAMADataset(self.args)
        else:
            self.data_test = LAMADataset(self.args, self.tokenizer)
            
        
        print(f"Filtered samples: {self.data_test.filter_count} items")
        print(f'All samples: {len(self.data_test)} items')
        print("finished data processing... costing {}s...".format(time.time() - now_time))

    def collate_fn(self, items):
        inputs = [item[1] for item in items]
        if 'roberta' in str(self.tokenizer.__class__):
            prefix = ' '
        else:
            prefix = ''
        outputs = [prefix+item[2] for item in items]
        # self.args.max_seq_length
        inputs_tokenized = self.tokenizer(inputs, padding='longest', max_length=512, truncation=True, return_tensors="pt")
        if inputs_tokenized.input_ids.shape[1] == 512:
            del_list = []
            for idx, input in enumerate(inputs_tokenized.input_ids):
                mask_idx = (input == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
                if len(mask_idx[0]) == 0:
                    del_list.append(idx)
            del_list.reverse()
            for idx in del_list:
                del inputs[idx]
                del outputs[idx]
            inputs_tokenized = self.tokenizer(inputs, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            outputs_tokenized = self.tokenizer(outputs, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=False)
        else:
            outputs_tokenized = self.tokenizer(outputs, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=False)
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        labels, _ = outputs_tokenized.input_ids, outputs_tokenized.attention_mask
        # print(self.tokenizer.decode(input_ids[0]))
        # print(self.tokenizer.decode(labels[0]))
        # exit(1)
        
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, batch_data=items)

    def collate_fn_for_pelt(self, items):
        if 'roberta' in str(self.tokenizer.__class__):
            prefix = ' '
        else:
            prefix = ''
        sub_labels = [item[0][0] for item in items]
        inputs = [[item[1]] for item in items]
        outputs = [prefix+item[2] for item in items]
        sub_ids = [item[3] for item in items]
        input_ids_list, attention_mask_list, entity_embeddings_list, entity_position_ids_list, masked_indices_list = self.pelt.get_batch(inputs, sub_labels=sub_labels, sub_ids=sub_ids)
        outputs_tokenized = self.tokenizer(outputs, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=False)
        labels, _ = outputs_tokenized.input_ids, outputs_tokenized.attention_mask
        return dict(input_ids=input_ids_list,
                     attention_mask=attention_mask_list,
                     masked_indices_list=masked_indices_list,
                     entity_embeddings=entity_embeddings_list,
                     entity_position_ids=entity_position_ids_list,
                     labels=labels, 
                     batch_data=items)

    def train_dataloader(self):
        raise NotImplementedError("Please use LAMA to test model...")

    def val_dataloader(self):
        raise NotImplementedError("Please use LAMA to test model...")
    
    def test_dataloader(self, relation: str = None):
        self.data_test.set_relation(relation)
        if self.args.pelt:
            return DataLoader(self.data_test, sampler=LAMASampler(self.data_test, self.tokenizer), collate_fn=self.collate_fn_for_pelt, 
                batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, pin_memory=True, drop_last=False)
        return DataLoader(self.data_test, sampler=LAMASampler(self.data_test, self.tokenizer), collate_fn=self.collate_fn, 
                batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, pin_memory=True, drop_last=False)

VOCAB_SIZE = 85078
class KGRECDataModule(BaseKGRECDataModule):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False) \
                        if tokenizer == None else tokenizer
    
    @staticmethod
    def add_to_argparse(parser):
        BaseKGRECDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)

        parser.add_argument("--use_pretrain", type=int, default=1)
        parser.add_argument("--max_item_length", type=int, default=200)
        parser.add_argument("--max_predictions_per_seq", type=int, default=20)
        parser.add_argument("--masked_lm_prob", type=float, default=0.2)
        parser.add_argument("--dupe_factor", type=int, default=10)
        parser.add_argument("--prop_sliding_window", type=float, default=0.5)
        parser.add_argument("--negative_item", type=int, default=100)
        return parser

    def setup(self, stage):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = KGRECDataset(self.args, mode='train')
            self.data_valid = KGRECDataset(self.args, mode='valid')
        else:
            self.data_test = KGRECDataset(self.args, mode='test')
        
        print("finished data processing... costing {}s...".format(time.time() - now_time))

    def collate_fn(self, items, mode):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label = []
        mask_pos = []
        negs = []
        for item_idx,item in enumerate(items):
            if mode == 'train':
                assert len(item.input) <= self.args.max_item_length
                label_ = []
                num_desc = len(item.input) - len(item.mask)
                input_list = item.input.copy()
                for msk in item.mask:
                    label_.append(item.input[msk])
                    input_list[msk] = self.tokenizer.mask_token

                if self.args.use_pretrain:
                    token_list = []
                    for it in input_list:
                        if it == self.tokenizer.mask_token:
                            token_list.append(it)
                        else:
                            token_list.extend([self.tokenizer.pad_token]*2)
                    input_ = self.tokenizer(f'Here is the watch history of a user: {" ".join(token_list)}.',
                        padding='longest', truncation="longest_first", max_length=self.args.max_seq_length*2,
                            )
                    cnt = 0
                    cnt_mask = 0
                    flag = -1
                    pos = []
                    for i in range(len(input_.input_ids)):
                        if input_.input_ids[i] == self.tokenizer.pad_token_id and cnt < num_desc*2:
                            if cnt & 1:
                                assert flag != -1
                                input_.input_ids[i] = len(self.tokenizer) + flag
                                assert input_.input_ids[i] < VOCAB_SIZE
                                cnt += 1
                                flag = -1
                            else:
                                assert flag == -1
                                assert input_list[(cnt >> 1) + cnt_mask] != self.tokenizer.mask_token
                                input_.input_ids[i] = len(self.tokenizer) + self.num_item + input_list[(cnt >> 1) + cnt_mask]
                                assert input_.input_ids[i] < VOCAB_SIZE
                                flag = input_list[(cnt >> 1) + cnt_mask]
                                cnt += 1
                        elif input_.input_ids[i] == self.tokenizer.mask_token_id:
                            pos.append(i)
                            if random.random() > 0.9:
                                try:
                                    input_.input_ids[i] = len(self.tokenizer) + self.num_item + label_[cnt_mask]
                                except:
                                    print(len(label_),cnt_mask)
                                    print(input_list)
                                    print(label_)
                                assert input_.input_ids[i] < VOCAB_SIZE
                            elif random.random() > 0.8:
                                input_.input_ids[i] = len(self.tokenizer) + self.num_item + random.choice(range(self.num_item))
                                assert input_.input_ids[i] < VOCAB_SIZE
                            cnt_mask += 1
                    assert len(pos) == len(label_)
                else:
                    token_list = []
                    for it in input_list:
                        if it == self.tokenizer.mask_token:
                            token_list.append(it)
                        else:
                            token_list.append(self.tokenizer.sep_token)
                    input_ = self.tokenizer(f'Here is the watch history of a user: {" ".join(token_list)}.',
                        padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
                            )
                    cnt = 0
                    cnt_mask = 0
                    pos = []
                    for i in range(len(input_.input_ids)):
                        if input_.input_ids[i] == self.tokenizer.pad_token_id and cnt < num_desc:
                            input_.input_ids[i] = len(self.tokenizer) + input_list[cnt]
                        elif input_.input_ids[i] == self.tokenizer.mask_token_id:
                            pos.append(i)
                            if random.random() > 0.9:
                                input_.input_ids[i] = label_[cnt_mask]
                            elif random.random() > 0.8:
                                input_.input_ids[i] = len(self.tokenizer) + random.choice(range(self.num_item))
                            cnt_mask += 1
                    assert len(pos) == len(label_)
                mask_pos.append(pos)
            else:
                assert len(item.input) <= self.args.max_item_length
                label_ = []
                num_desc = len(item.input) - len(item.mask)
                input_list = item.input
                for msk in item.mask:
                    label_.append(item.input[msk])
                    input_list[msk] = self.tokenizer.mask_token
                assert len(label_) == 1
                if self.args.use_pretrain:
                    token_list = []
                    for it in input_list:
                        if it == self.tokenizer.mask_token:
                            token_list.append(it)
                        else:
                            token_list.extend([self.tokenizer.pad_token]*2)
                    input_ = self.tokenizer(f'Here is the watch history of a user: {" ".join(token_list)}.',
                        padding='longest', truncation="longest_first", max_length=self.args.max_seq_length*2,
                            )
                    cnt = 0
                    flag = -1
                    for i in range(len(input_.input_ids)):
                        if input_.input_ids[i] == self.tokenizer.pad_token_id and cnt < num_desc*2:
                            if cnt & 1:
                                assert flag != -1
                                input_.input_ids[i] = len(self.tokenizer) + flag
                                cnt += 1
                                flag = -1
                            else:
                                assert flag == -1
                                assert input_list[cnt >> 1] != self.tokenizer.mask_token
                                input_.input_ids[i] = len(self.tokenizer) + self.num_item + input_list[cnt >> 1]
                                flag = input_list[cnt >> 1]
                                cnt += 1
                else:
                    token_list = []
                    for it in input_list:
                        if it == self.tokenizer.mask_token:
                            token_list.append(it)
                        else:
                            token_list.append(self.tokenizer.sep_token)
                    input_ = self.tokenizer(f'Here is the watch history of a user: {" ".join(token_list)}.',
                        padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
                            )
                    cnt = 0
                    for i in range(len(input_.input_ids)):
                        if input_.input_ids[i] == self.tokenizer.pad_token_id and cnt < num_desc:
                            input_.input_ids[i] = len(self.tokenizer) + input_list[cnt]
                negs.append(item.neg)

                
            input_ids.append(input_.input_ids)
            attention_mask.append(input_.attention_mask)
            token_type_ids.append(input_.token_type_ids)
            label.extend(label_)

        # print('input_ids', input_ids[0])
        # print('decode', self.tokenizer.decode(input_ids[0]))
        # print('attention_mask', attention_mask[0])
        # print('token_type_ids', token_type_ids[0])
        # print('label', label[0])
        
        features =  dict(input_ids=input_ids, attention_mask=attention_mask, 
                token_type_ids=token_type_ids)
        
        features = self.tokenizer.pad(
            features,
            padding='longest',
            return_tensors="pt"
        )
        features.update(dict(label=torch.tensor(np.array(label))))
        if mode == 'train':
            # print(mask_pos)
            mask_ = torch.zeros_like(features.input_ids)
            for idx in range(len(mask_pos)):
                for j in mask_pos[idx]:
                    mask_[idx][j] = 1
            features.update(dict(mask_pos=mask_))
            #print('mask_pos', mask_[0])
        else:
            features.update(dict(negs=torch.tensor(np.array(negs))))
            # print('negs', negs[0])

        # print('input_ids', features['input_ids'].shape)
        # print('attention_mask', features['attention_mask'].shape)
        # print('token_type_ids', features['token_type_ids'].shape)
        # print('label', features['label'].shape)
        # print('mask_pos', features['mask_pos'].shape)
        return features
    
        
class KGRECPretrainDataModule(KGRECDataModule):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False) \
                        if tokenizer == None else tokenizer
    
    @staticmethod
    def add_to_argparse(parser):
        BaseKGCDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=32)
        return parser

    def setup(self, stage):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = PretrainKGRECDataset(self.args)
            #self.data_val = PretrainKGRECDataset(self.args)
        else:
            self.data_test = PretrainKGRECDataset(self.args)

        # self.filter_hr_to_t = defaultdict(list)

        # with open(f"dataset/{self.args.dataset}/movies.txt") as file:
        #     for line in file.readlines():
        #         h, r, t = lmap(int,line.strip().split('\t'))
        #         self.filter_hr_to_t[(h,r)].append(t)
        
        # self.filter_hr_to_t = {k: list(set(v)) for k, v in self.filter_hr_to_t.items()}
        # max_filter_ent = max(max([len(_) for _ in self.filter_hr_to_t.values()]), max([len(_) for _ in self.filter_tr_to_h.values()]))
        # print("=== max filter ent {} ===".format(max_filter_ent))
            
        print("finished data processing... costing {}s...".format(time.time() - now_time))

    def collate_fn(self, items, mode):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label = []
        for item_idx,item in enumerate(items):
            e = item.e
            words = item.text.split()
            st = 0
            input_ = self.tokenizer(f"The description of {self.tokenizer.mask_token} is {' '.join(words[st: min(st+200, len(words))])}",
                padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
                    )
            input_ids.append(input_.input_ids)
            attention_mask.append(input_.attention_mask)
            token_type_ids.append(input_.token_type_ids)
            label.append(e)


            # input_ = self.tokenizer(f"The description is {' '.join(words[st: min(st+150, len(words))])} {self.tokenizer.mask_token}.",
            #     padding='longest', truncation="longest_first", max_length=self.args.max_seq_length,
            #         )
            # input_ids.append(input_.input_ids)
            # attention_mask.append(input_.attention_mask)
            # token_type_ids.append(input_.token_type_ids)
            
            # label.append(e)
        
        # print('input_ids', input_ids[0])
        # print('decode', self.tokenizer.decode(input_ids[0]))
        # print('attention_mask', attention_mask[0])
        # print('token_type_ids', token_type_ids[0])
        # print('label', label[0])

        features =  dict(input_ids=input_ids, attention_mask=attention_mask, 
                token_type_ids=token_type_ids)
        
        features = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=self.args.max_seq_length,
            return_tensors="pt"
        )
        features.update(dict(label=torch.tensor(label)))
        return features

    def val_dataloader(self):
        pass
