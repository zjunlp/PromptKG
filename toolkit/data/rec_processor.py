import contextlib
from hashlib import new
import sys
from typing import Dict, List

from collections import Counter, defaultdict
from multiprocessing import Pool
from tokenizers import Tokenizer
from torch.utils.data import Dataset, Sampler, IterableDataset
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import os
import ujson
import random
import json
import torch
import copy
import numpy as np
import pickle
from tqdm import tqdm
from dataclasses import dataclass, asdict, replace
import inspect

from transformers.models.auto.tokenization_auto import AutoTokenizer

@dataclass
class PretrainExample:
    e: int = None
    text: str = None

@dataclass
class Example:
    input : list
    mask : list
    neg: list = None


class PretrainKGRECDataset(Dataset):
    def __init__(self, 
                args,
                mode=None
                ):
        super().__init__()
        dataset_name = args.dataset
        self.args = args
        self.mode = mode
        self.data = self.loadData()

    def loadData(self):
        data = []
        with open(f"./dataset/{self.args.dataset}/item2text.txt") as file:
            for line in file.readlines():
                e, text = line.strip().split("\t")
                data.append(PretrainExample(e=int(e),text=text))
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class KGRECDataset(Dataset):
    def __init__(self, 
                args,
                mode=None
                ):
        super().__init__()
        dataset_name = args.dataset
        self.args = args
        self.mode = mode
        self.max_item_length = self.args.max_item_length
        self.max_predictions_per_seq = self.args.max_predictions_per_seq
        self.masked_lm_prob = self.args.masked_lm_prob
        self.dupe_factor = self.args.dupe_factor
        self.prop_sliding_window = self.args.prop_sliding_window
        self.negative_item = self.args.negative_item
        self.data = self.loadData()

    def loadData(self):
        if True and os.path.exists(f"./dataset/{self.args.dataset}/{self.mode}.pkl") and os.path.exists(f"./dataset/{self.args.dataset}/popularity.pkl"):
            with open(f"./dataset/{self.args.dataset}/{self.mode}.pkl", 'rb') as fp:
                data = pickle.load(fp)
            print(f"Load {self.mode} data from ./dataset/{self.args.dataset}/{self.mode}.pkl.")
            if self.mode != 'train':
                with open(f"./dataset/{self.args.dataset}/popularity.pkl", 'rb') as fp:
                    self.popularity = pickle.load(fp)
                print(f"Load popularity data from ./dataset/{self.args.dataset}/popularity.pkl.")
        else:
            train = dict()
            with open(f"./dataset/{self.args.dataset}/train.txt") as file:
                for line in file.readlines():
                    uid, li = line.split('\t')
                    uid = int(uid.strip())
                    li = list(map(int, li.strip().split(',')))
                    train[uid] = li
            
            with open(f"./dataset/{self.args.dataset}/train.pkl", 'wb') as fp:
                pickle.dump(train,fp)

            valid = dict()
            with open(f"./dataset/{self.args.dataset}/valid.txt") as file:
                for line in file.readlines():
                    uid, li = line.split('\t')
                    uid = int(uid.strip())
                    li = list(map(int, li.strip().split(',')))
                    train[uid].extend(li)
                    valid[uid] = train[uid]
                
            popularity_dict = defaultdict(int)
            self.popularity = []
            for v in train.values():
                for i in v:
                    popularity_dict[i] += 1

            for i in range(len(popularity_dict)):
                self.popularity.append(popularity_dict[i])

            with open(f"./dataset/{self.args.dataset}/popularity.pkl", 'wb') as fp:
                pickle.dump(self.popularity,fp)

            test = dict()
            with open(f"./dataset/{self.args.dataset}/test.txt") as file:
                for line in file.readlines():
                    uid, li = line.split('\t')
                    uid = int(uid.strip())
                    li = list(map(int, li.strip().split(',')))
                    test[uid] = train[uid]
                    test[uid].extend(li)

            train = self.mask_data(train, 'train')
            with open(f"./dataset/{self.args.dataset}/train.pkl", 'wb') as fp:
                pickle.dump(train,fp)

            valid = self.mask_data(valid, 'valid')
            with open(f"./dataset/{self.args.dataset}/valid.pkl", 'wb') as fp:
                pickle.dump(valid,fp)

            test = self.mask_data(test, 'test')
            with open(f"./dataset/{self.args.dataset}/test.pkl", 'wb') as fp:
                pickle.dump(test,fp)

            if self.mode == 'train':
                data = train
            elif self.mode == 'valid':
                data = valid
            elif self.mode == 'test':
                data = test
            else:
                raise RuntimeError
        
        data = self.to_example(data)

        return data

    def to_example(self, data):
        newdata = []
        for item in data:
            if 'neg' in item.keys():
                newdata.append(Example(input=item['input'],mask=item['mask'],neg=item['neg']))
            else:
                newdata.append(Example(input=item['input'],mask=item['mask']))
        return newdata

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def mask_data(self, data, mode):
        newdata = []
        raw = []
        if mode == 'train':
            for _, v in tqdm(data.items(),desc='Process instance'):
                if len(v) <= self.max_item_length:
                    raw.append(v)
                else:
                    sliding_step = (int)(self.prop_sliding_window * self.max_item_length)
                    beg_idx = list(range(len(v)-self.max_item_length, 0, -sliding_step))
                    beg_idx.append(0)
                    raw.extend([v[i:i + self.max_item_length] for i in beg_idx[::-1]])
            for it in tqdm(raw,desc='Mask instance'):
                for _ in range(self.dupe_factor):
                    mask_pos = []
                    for idx, i in enumerate(it):
                        if random.random() < self.masked_lm_prob:
                            mask_pos.append(idx)
                    newdata.append(dict(input=it, mask=mask_pos))
                newdata.append(dict(input=it, mask=[-1]))
        else:
            for _, v in tqdm(data.items(),desc='Process instance'):
                if len(v) <= self.max_item_length:
                    raw.append(v)
                else:
                    raw.append(v[-self.max_item_length:])
            item_num = len(self.popularity)
            
            for it in tqdm(raw,desc='Mask instance'):
                choice_idx = list(set(range(item_num))-set(it))
                choice_idx = sorted(choice_idx)
                popularity= list(np.array(self.popularity)[choice_idx])
                sump = sum(popularity)
                popularity_prob = list(map(lambda x: x/sump, popularity))
                neg_item = list(np.random.choice(choice_idx,100,p=popularity_prob,replace=False))
                newdata.append(dict(input=it, mask=[-1], neg=neg_item))
        
        return newdata
            
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--max_item_length", type=int, default=200)
        parser.add_argument("--max_predictions_per_seq", type=int, default=20)
        parser.add_argument("--masked_lm_prob", type=float, default=0.2)
        parser.add_argument("--dupe_factor", type=int, default=10)
        parser.add_argument("--prop_sliding_window", type=float, default=0.5)
        parser.add_argument("--negative_item", type=int, default=100)
        return parser


if __name__ == "__main__":
    class Config(dict):
        def __getattr__(self, name):
            return self.get(name)

        def __setattr__(self, name, val):
            self[name] = val
    args_ = dict(
        dataset='ml20m',
        max_item_length=200,
        max_predictions_per_seq=20,
        masked_lm_prob=0.2,
        dupe_factor=10,
        prop_sliding_window=0.5,
        negative_item=100
    )
    args = Config(args_)
    dataset = KGRECDataset(args, 'test')
    print(dataset[:3])