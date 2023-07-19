import contextlib
import sys
from typing import Dict, List

from collections import Counter
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
            inputs.append(args.prompt+line[0])
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
                tokenizer,
                mode=None
                ):
        # self.dataset_name = args.dataset if args is not None else 'LAMA'
        self.subdataset_list = ['Google_RE', 'Squad', 'TREx', 'ConceptNet']
        self.subdataset = [args.lamadataset] if args.lamadataset is not None \
                        else self.subdataset_list
        self.data_dir = lambda path: os.path.join(f'./dataset/LAMA',path)

        self.template = {"place_of_birth": "[X] was born in [Y] .", 
                        "date_of_birth": "[X] (born [Y]).", 
                        "place_of_death": "[X] died in [Y] ."}
        
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.relation = None
        
        self.rel_label = dict()
        self.get_label = lambda id : self.rel_label[id].replace(' ', '_') if id.startswith('P') else id
        if 'TREx' in self.subdataset:
            with open(self.data_dir('relations.jsonl')) as fp:
                for i in fp:
                    line = ujson.loads(i)
                    self.rel_label[line['relation']] = line['label']
                    self.template.update({line['relation']: line['template']})
        self.filter_count = 0
        self.valid_label = set()
        self.all_data, self.data = self.load_data()

    def set_relation(self, relation):
        assert relation in self.data.keys() or relation == None
        self.relation = relation

    def process_sample(self, hrt, masked_sentence, label):
        # TODO: split the sentences
        mask_token = self.tokenizer.special_tokens_map['mask_token']
        sep_token = self.tokenizer.special_tokens_map['sep_token']
        masked_sentence = masked_sentence.replace('[MASK]', mask_token)
        masked_sentence = masked_sentence.replace('[SEP]', sep_token)
        mask_count = masked_sentence.count(mask_token)
        # beyond the max length of BertEncoder (more than 500)
        sentence_filter = (mask_count != 1) or (len(masked_sentence.split()) > 500)
        if sentence_filter:
            return (None, None, None)
        # islowercase = 'uncased' in self.tokenizer.name_or_paths
        # if islowercase:
        #     hrt = tuple(map(lambda x: x if x == None else x.lower(), hrt))
        #     masked_sentences = masked_sentences.lower
        #     for token in self.tokenizer.special_tokens.values():
        #         masked_sentences.replace(token.lower(), token)
        #     labels = labels.lower()
        # if label in self.valid_label:
        #     return (hrt, masked_sentence, label)
        if 'roberta' in str(self.tokenizer.__class__):
            prefix = ' '
        else:
            prefix = ''
        label_tokens = self.tokenizer.tokenize(prefix+label)
        label_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)
        if not label_ids:
            return (None, None, None)
        vocab = list(self.tokenizer.get_vocab())
        recostructed_word = " ".join([vocab[x] for x in label_ids]).strip('Ġ').strip()
        # print('recostructed_word    ',recostructed_word)
        # exit(1)
        if recostructed_word != label and recostructed_word != label.lower():
            return (None, None, None)
        # new_label = self.tokenizer.decode(label_ids)
        # if (new_label != label and new_label != label.lower()) or len(label.split()) != 1:
        #     return (None, None, None)

        self.valid_label.add(label)
        return (hrt, masked_sentence, label)

    @staticmethod
    def parse_template(template, subject_label):
        SUBJ_SYMBOL = "[X]"
        OBJ_SYMBOL = "[Y]"
        template = template.replace(SUBJ_SYMBOL, subject_label)
        template = template.replace(OBJ_SYMBOL, "[MASK]")
        return [template]

    def load_data(self):
        all_data = []
        data = dict()
        for subdataset in self.subdataset:
            print(f"Loading {subdataset}...")
            for file in tqdm(os.listdir(self.data_dir(f'{subdataset}'))):
                if not '.jsonl' in file:
                    continue
                with open(self.data_dir(f'{subdataset}/{file}')) as fp:
                    _file = file.split('.')[0].rsplit('_', 1)[0]
                    key_name = subdataset + '-'+self.get_label(_file)
                    template = self.template[_file] if _file in self.template.keys() else None
                    data[key_name] = []
                    for i in fp:
                        line = ujson.loads(i)
                        hrt, masked_sentences, labels, sub_id = self.parse_line(line, subdataset, template)
                        if hrt == None:
                            self.filter_count += 1
                            continue
                        assert len(masked_sentences) == len(labels)
                        line_data = []
                        for idx in range(len(masked_sentences)):
                            hrt, masked_sentence, label = self.process_sample(hrt, masked_sentences[idx], labels[idx])
                            if hrt == None:
                                self.filter_count += 1
                                break
                            line_data.append(dict(hrt=hrt, masked_sentence=masked_sentence, label=label, sub_id=sub_id))
                        data[key_name].extend(line_data)
                        all_data.extend(line_data)
        return (all_data, data, )

    def parse_line(self, data_line, dataset_type: str, template: str):
        if dataset_type == 'Google_RE':
            hrt = (data_line['sub_label'], data_line['pred'], data_line['obj_label'])
            masked_sentences = [' [SEP] '.join(data_line['masked_sentences'])]
            if template:
                masked_sentences = self.parse_template(template, data_line['sub_label'])
            labels = [data_line['obj_label']]
            num_no = 0
            num_yes = 0
            for x in data_line["judgments"]:
                if x["judgment"] == "yes":
                    num_yes += 1
                else:
                    num_no += 1
            if num_no > num_yes:
                return (None, None, None, None)
        elif dataset_type == 'Squad':
            hrt = (None, None, None)
            masked_sentences = data_line['masked_sentences']
            labels = [data_line['obj_label']]
        elif dataset_type == 'TREx':
            hrt = (data_line['sub_label'], self.rel_label[data_line['predicate_id']], data_line['obj_label'])
            masked_sentences_dict = dict()
            for i in data_line['evidences']:
                # obj_surface may has more than 1 token
                # masked_sentences_dict[i['masked_sentence']] = i['obj_surface']
                masked_sentences_dict[i['masked_sentence']] = data_line['obj_label']
            masked_sentences = []
            labels = []
            for k, v in masked_sentences_dict.items():
                masked_sentences.append(k)
                labels.append(v)
            if template:
                masked_sentences = self.parse_template(template, data_line['sub_label'])
                labels = [data_line['obj_label']]
        elif dataset_type == 'ConceptNet':
            if 'sub_label' in data_line.keys():
                hrt = (data_line['sub_label'], data_line['pred'], data_line['obj_label'])
            else:
                return (None, None, None, None)
                #hrt = (data_line['sub'], data_line['pred'], data_line['obj'])
            masked_sentences = data_line['masked_sentences']
            labels = [data_line['obj_label']]
        else:
            raise NotImplementedError
        
        assert len(masked_sentences) < 2
        # if len(masked_sentences[0].split()) > 100:
        #     return (None, None, None)
        if 'sub_uri' in data_line.keys():
            sub_id = data_line['sub_uri']
        else:
            sub_id = None 
        return (hrt, masked_sentences, labels, sub_id)

    def __len__(self):
        if self.relation == None:
            return len(self.all_data)
        else:
            return len(self.data[self.relation])
    
    def __getitem__(self, index):
        if self.relation == None:
            item = self.all_data[index]
        else:
            item = self.data[self.relation][index]
        hrt = item['hrt']
        masked_sentence = item['masked_sentence']
        label = item['label']
        sub_id = item['sub_id']
        return (hrt, masked_sentence, label, sub_id)

class LAMASampler(Sampler):
    def __init__(self, data_source, tokenizer) -> None:
        self.data_source = data_source
        self.tokenizer = tokenizer

    def __iter__(self):
        index_sorted = sorted(
            range(len(self.data_source)), key=lambda k: len(self.data_source[k][1].split()), reverse=True
        )
        return iter(index_sorted)

    def __len__(self) -> int:
        return len(self.data_source)

import pandas as pd
class CommonSenseDataset(Dataset):
    def __init__(self, args, mode) -> None:
        super().__init__()
        self.args = args
        self.mode = mode
        self.data = self.loadData()

    def loadData(self):
        data = []
        # warning, csv for common sense dataset
        data = pd.read_csv(f"./dataset/{self.args.dataset}/{self.mode}.csv")
        tmp_data = []
        hrt = ['head', 'relation', 'tail']
        for _ in hrt:
            data[_] = data[_].astype(str)
        for _, d in data.iterrows():
            tmp_data.append([d['head'], d['relation'], d['tail'], int(d['label'])])
        data = tmp_data
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    class ARG(object):
        def __init__(self) -> None:
            self.lamadataset = 'ConceptNet'
    args = ARG()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=False)
    dataset = LAMADataset(args, tokenizer)
    print(len(dataset))
