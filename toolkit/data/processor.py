import contextlib
import sys
from typing import Dict, List

from collections import Counter
from multiprocessing import Pool
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
"""
这个文件负责处理从
1. load dataset file into object
2. process the object into batch tensor
"""
@dataclass
class Example:
    """
    hr : list , h_index, r_index
    t : int, t_index
    """
    hr: list
    t: int
    inverse: bool

@dataclass
class PretrainExample:
    e: int

class KGCDataset(Dataset):
    def __init__(self, 
                args,
                mode
                ):
        super().__init__()
        dataset_name = args.dataset
        self.args = args
        # self.data = self.loadData(filename, max_points)
        self.mode = mode
        self.data = self.loadData()

    def loadData(self):
        data = []
        with open(f"./dataset/{self.args.dataset}/{self.mode}.tsv") as file:
            for line in file.readlines():
                h,r,t = list(map(int,line.strip().split()))
                data.append(Example(hr=(h,r),t=t, inverse=False))
                data.append(Example(hr=(t,r),t=h, inverse=True))
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class PretrainKGCDataset(Dataset):
    def __init__(self, 
                args,
                mode
                ):
        super().__init__()
        dataset_name = args.dataset
        self.args = args
        # self.data = self.loadData(filename, max_points)
        self.mode = mode
        self.data = self.loadData()

    def loadData(self):
        data = []
        with open(f"./dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                e, text = line.strip().split("\t")
                data.append(int(e))
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class KGT5Dataset(Dataset):
    def __init__(self, 
                args,
                tokenizer,
                mode
                ):
        super().__init__()
        dataset_name = args.dataset
        self.tokenizer = tokenizer
        max_points = -1
        # self.data = self.loadData(filename, max_points)
        self.splits = dict()
        self.data = self.loadData(f"dataset/{dataset_name}/{mode}.txt", -1)
        # self.data = self.loadData(f"dataset/{dataset_name}/dev.txt", -1)
        self.mode = mode
        # self.splits["train"] = self.loadData(f"dataset/{dataset_name}/train.txt", max_points)
        # self.splits["valid"] = self.loadData(f"dataset/{dataset_name}/valid.txt", max_points)
        # self.splits["test"] = self.loadData(f"dataset/{dataset_name}/test.txt", max_points)
        # self.entity_strings = self.load_entity_strings(os.path.join("dataset", dataset_name, "entity_strings.txt"))
        # self.tokenized_entities = self.tokenizer(self.entity_strings, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        # self.entity_string_to_id = dict(zip(self.entity_strings, torch.arange(len(self.entity_strings)).tolist()))

    def split(self, split: str) -> Dict[List[str], List[str]]:
        return self.splits[split]

    def numLines(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    
    def loadData(self, filename, max_points):
        file_len = self.numLines(filename)
        f = open(filename, 'r')
        inputs = []
        outputs = []
        h = []
        target_entity_ids = []
        for i in tqdm(range(file_len)):
            if i == max_points:
                break
            line = f.readline()
            if line[-1] == '\n':
                line = line[:-1]
            line = line.split('\t')
            inputs.append(line[0])
            outputs.append(line[1])
            h.append((int(line[2]), int(line[3])))
            target_entity_ids.append(int(line[4]))
            
        data = {'inputs': inputs, 'outputs': outputs, 'hr_pair': h, "target_entity_ids": target_entity_ids}
        return data

    @staticmethod
    def load_entity_strings(filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        return lines


    def __len__(self):
        return len(self.data['inputs'])

    def __getitem__(self, index):
        data = self.data
        input = data['inputs'][index]
        output = data['outputs'][index]
        hr_pair = data['hr_pair'][index]
        target_entity_id = data['target_entity_ids'][index]
        return input, output, hr_pair, target_entity_id


    

    def _collate_fn_new(self, items):
        inputs = [item[0] for item in items]
        outputs = [item[1] for item in items]
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, max_length=128, return_tensors="pt")
        outputs_tokenized = self.tokenizer(outputs, padding=True, truncation=True, max_length=32, return_tensors="pt")
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        labels, labels_attention_mask = outputs_tokenized.input_ids, outputs_tokenized.attention_mask
        # for labels, set -100 for padding
        if self.mode == "train": labels[labels==0] = -100
        # labels = -100 * torch.ones(labels.shape, dtype=torch.long)
        return input_ids, attention_mask, labels, labels_attention_mask


    def _collate_eval(self, items):
        inputs = [item[0] for item in items]
        target_text = [item[1] for item in items]
        # inputs_tokenized = self.tokenizer(inputs, return_tensors="pt")
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        # inputs_tokenized = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        # print(inputs_tokenized.input_ids)
        # print(inputs_tokenized.attention_mask)
        
        # print(inputs_tokenized.attention_mask)
        return inputs_tokenized.input_ids, inputs_tokenized.attention_mask, target_text

    def _collate_eval_2(self, items):
        inputs = [item[0] for item in items]
        target_text = [item[1] for item in items]
        # inputs_tokenized = self.tokenizer(inputs, return_tensors="pt")
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        # inputs_tokenized = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        # print(inputs_tokenized.input_ids)
        # print(inputs_tokenized.attention_mask)

        # print(inputs_tokenized.attention_mask)
        return inputs_tokenized.input_ids, inputs_tokenized.attention_mask, target_text, inputs

    def tokenizedToText(self, arr):
        return ''.join(self.tokenizer.convert_ids_to_tokens(arr))


class LAMADataset(Dataset):
    def __init__(self, 
                args,
                mode
                ):
        self.dataset_name = args.dataset
        self.subdataset = args.subdataset
        self.subdataset_list = ['Google_RE', 'Squad', 'TREx', 'ConceptNet']
        self.args = args
        self.mode = mode

        self.rel_label = dict()
        if self.subdataset in [None, 'TREx']:
            with open(f'./dataset/{self.dataset_name}/relations.jsonl') as fp:
                for i in fp:
                    line = ujson.loads(i)
                    self.rel_label[line['relation']] = line['label']

        self.all_data, self.data = self.loadData()

        self.lama_full = []
        if self.subdataset == None:
            for v in self.all_data.values():
                self.lama_full += v

    def loadData(self):
        all_data = []
        data = dict()
        if self.subdataset == None:
            all_data = dict()
            for subdataset in self.subdataset_list:
                data[subdataset] = dict()
                all_data[subdataset] = []
                for file in os.listdir(f'./dataset/{self.dataset_name}/{subdataset}'):
                    with open(f'./dataset/{self.dataset_name}/{subdataset}/{file}') as fp:
                        data[subdataset][file] = []
                        for i in fp:
                            line = ujson.loads(i)
                            hrt, masked_sentences, labels = self.parse_line(line, subdataset)
                            assert len(masked_sentences) == len(labels)
                            line_data = []
                            for idx in range(len(masked_sentences)):
                                line_data.append(dict(hrt=hrt, masked_sentence=masked_sentences[idx], label=labels[idx]))
                            data[subdataset][file].extend(line_data)
                            all_data[subdataset].extend(line_data)
        else:
            for file in os.listdir(f'./dataset/{self.dataset_name}/{self.subdataset}'):
                with open(f'./dataset/{self.dataset_name}/{self.subdataset}/{file}') as fp:
                    data[file] = []
                    for i in fp:
                        line = ujson.loads(i)
                        hrt, masked_sentences, labels = self.parse_line(line, self.subdataset)
                        assert len(masked_sentences) == len(labels)
                        line_data = []
                        for idx in range(len(masked_sentences)):
                            line_data.append(dict(hrt=hrt, masked_sentence=masked_sentences[idx], label=labels[idx]))
                        data[file].extend(line_data)
                        all_data.extend(line_data)

        return (all_data, data, )

    def parse_line(self, data_line, dataset_type: str):
        if dataset_type == 'Google_RE':
            hrt = (data_line['sub_label'], data_line['pred'], data_line['obj_label'])
            masked_sentences = [''.join(data_line['masked_sentences'])]
            labels = [data_line['obj_label']]
        elif dataset_type == 'Squad':
            hrt = (None, None, None)
            masked_sentences = data_line['masked_sentences']
            labels = [data_line['obj_label']]
        elif dataset_type == 'TREx':
            hrt = (data_line['sub_label'], self.rel_label[data_line['predicate_id']], data_line['obj_label'])
            masked_sentences_dict = dict()
            for i in data_line['evidences']:
                masked_sentences_dict[i['masked_sentence']] = i['obj_surface']
            masked_sentences = []
            labels = []
            for k, v in masked_sentences_dict.items():
                masked_sentences.append(k)
                labels.append(v)
        elif dataset_type == 'ConceptNet':
            if 'sub_label' in data_line.keys():
                hrt = (data_line['sub_label'], data_line['pred'], data_line['obj_label'])
            else:
                hrt = (data_line['sub'], data_line['pred'], data_line['obj'])
            masked_sentences = data_line['masked_sentences']
            labels = [data_line['obj_label']]
        else:
            raise NotImplementedError
        return (hrt, masked_sentences, labels, )

    def __len__(self):
        if self.subdataset == None:
            return len(self.lama_full)
        else:
            return len(self.all_data)
    
    def __getitem__(self, index):
        if self.subdataset == None:
            item = self.lama_full[index]
        else:
            item = self.all_data[index]
        hrt = item['hrt']
        masked_sentence = item['masked_sentence']
        label = item['label']
        return (hrt, masked_sentence, label, )

if __name__ == "__main__":
    '''
    LAMA test
    '''
    class ARGs(object):
        def __init__(self) -> None:
            self.dataset = 'LAMA'
            self.subdataset = ['Google_RE', 'Squad', 'TREx', 'ConceptNet', None][3]
    os.chdir('../')
    args = ARGs()
    lama = LAMADataset(args, mode=None)
    print(len(lama))
    print(lama[0])