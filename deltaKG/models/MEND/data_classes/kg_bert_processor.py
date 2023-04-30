from re import DEBUG

import contextlib
import sys

from collections import Counter
from multiprocessing import Pool
from torch._C import HOIST_CONV_PACKED_PARAMS
from torch.utils.data import Dataset, Sampler, IterableDataset
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import os
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
from utils import EditBatchSampler, dict_to

# from models.utils import get_entity_spans_pre_processing


def lmap(a, b):
    return list(map(a, b))


def cache_results(_cache_fp, _refresh=False, _verbose=1):
    r"""
    cache_results是fastNLP中用于cache数据的装饰器。通过下面的例子看一下如何使用::

        import time
        import numpy as np
        from fastNLP import cache_results
        
        @cache_results('cache.pkl')
        def process_data():
            # 一些比较耗时的工作，比如读取数据，预处理数据等，这里用time.sleep()代替耗时
            time.sleep(1)
            return np.random.randint(10, size=(5,))
        
        start_time = time.time()
        print("res =",process_data())
        print(time.time() - start_time)
        
        start_time = time.time()
        print("res =",process_data())
        print(time.time() - start_time)
        
        # 输出内容如下，可以看到两次结果相同，且第二次几乎没有花费时间
        # Save cache to cache.pkl.
        # res = [5 4 9 1 8]
        # 1.0042750835418701
        # Read cache from cache.pkl.
        # res = [5 4 9 1 8]
        # 0.0040721893310546875

    可以看到第二次运行的时候，只用了0.0001s左右，是由于第二次运行将直接从cache.pkl这个文件读取数据，而不会经过再次预处理::

        # 还是以上面的例子为例，如果需要重新生成另一个cache，比如另一个数据集的内容，通过如下的方式调用即可
        process_data(_cache_fp='cache2.pkl')  # 完全不影响之前的‘cache.pkl'

    上面的_cache_fp是cache_results会识别的参数，它将从'cache2.pkl'这里缓存/读取数据，即这里的'cache2.pkl'覆盖默认的
    'cache.pkl'。如果在你的函数前面加上了@cache_results()则你的函数会增加三个参数[_cache_fp, _refresh, _verbose]。
    上面的例子即为使用_cache_fp的情况，这三个参数不会传入到你的函数中，当然你写的函数参数名也不可能包含这三个名称::

        process_data(_cache_fp='cache2.pkl', _refresh=True)  # 这里强制重新生成一份对预处理的cache。
        #  _verbose是用于控制输出信息的，如果为0,则不输出任何内容;如果为1,则会提醒当前步骤是读取的cache还是生成了新的cache

    :param str _cache_fp: 将返回结果缓存到什么位置;或从什么位置读取缓存。如果为None，cache_results没有任何效用，除非在
        函数调用的时候传入_cache_fp这个参数。
    :param bool _refresh: 是否重新生成cache。
    :param int _verbose: 是否打印cache的信息。
    :return:
    """

    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError(
                    "The function decorated by cache_results cannot have keyword `{}`."
                    .format(key))

        def wrapper(*args, **kwargs):
            my_args = args[0]
            mode = args[-1]
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath,
                                  str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose,
                                  int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True

            # model_name = my_args.model_name.split("/")[-1]
            # is_pretrain = my_args.pretrain
            # refresh = my_args.overwrite_cache

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = pickle.load(f)
                    if verbose == 1:
                        logger.info(
                            "Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError(
                            "The return value is None. Delete the decorator.")
                    with open(cache_filepath, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_


import argparse
import csv
import logging
import os
import random
import sys
import jsonlines

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# from torch.nn import CrossEntropyLoss, MSELoss
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_scoreclass

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 text_c=None,
                 label=None,
                 wrong_label=None,
                 target_label=None,
                 cor_head=None,
                 en=None,
                 rel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. list of entities
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.wrong_label = wrong_label
        self.target_label = target_label
        self.cor_head = cor_head
        self.en = en
        self.rel = rel  # rel id


@dataclass
class InputFeatures:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    label: torch.Tensor = None
    cor_head: torch.Tensor = 0
    en: torch.Tensor = 0
    rel: torch.Tensor = 0
    pos: torch.Tensor = 0


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file):
        with open(input_file, "r") as f:
            lines = []
            for line in f.readlines():
                tri = line.rstrip().split('\t')
                lines.append(tri)
        return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        if input_file.split('/')[-1] == 'stable.jsonl':
            with jsonlines.open(input_file) as f:
                lines = []
                for d in f:
                    lines.append((d['process'], d["triple"], d['label']))
            return lines
        elif input_file.split('/')[-2] == 'AddKnowledge':
            with jsonlines.open(input_file) as f:
                lines = []
                for d in f:
                    lines.append((d["label"], d['triples'], d['head']))
            return lines
        else:
            with jsonlines.open(input_file) as f:
                lines = []
                for d in f:
                    lines.append((d["ori"], d['cor'], d['label']))
                    # lines.append(d['cor'])
            return lines



import copy


def solve_get_knowledge_store(line, set_type="train", pretrain=1):
    """
    use the LM to get the entity embedding.
    Transductive: triples + text description
    Inductive: text description
    
    """
    examples = []

    head_ent_text = ent2text[line[0]]
    tail_ent_text = ent2text[line[2]]
    relation_text = rel2text[line[1]]

    i = 0

    a = tail_filter_entities["\t".join([line[0], line[1]])]
    b = head_filter_entities["\t".join([line[2], line[1]])]

    guid = "%s-%s" % (set_type, i)
    text_a = head_ent_text
    text_b = relation_text
    text_c = tail_ent_text

    # use the description of c to predict A
    examples.append(
        InputExample(guid=guid,
                     text_a="[PAD]",
                     text_b=text_b + "[PAD]",
                     text_c="[PAD]" + " " + text_c,
                     label=lmap(lambda x: ent2id[x], b),
                     real_label=ent2id[line[0]],
                     en=[ent2id[line[0]], rel2id[line[1]], ent2id[line[2]]],
                     rel=0))
    examples.append(
        InputExample(guid=guid,
                     text_a="[PAD]",
                     text_b=text_b + "[PAD]",
                     text_c="[PAD]" + " " + text_a,
                     label=lmap(lambda x: ent2id[x], b),
                     real_label=ent2id[line[2]],
                     en=[ent2id[line[0]], rel2id[line[1]], ent2id[line[2]]],
                     rel=0))
    return examples


def solve(line, data_dir = None, set_type="train", pretrain=1, task_name=None):
    if set_type == 'memory':
        _, line, label = line[0], line[1], line[2]
        head = 1 if line[0] == label else 0
        label_text = ent2text[label]
        head_ent_text = ent2text[line[0]]
        tail_ent_text = ent2text[line[2]]
        relation_text = rel2text[line[1]]
    elif task_name == 'add':
        label, line, head = line[0], line[1], line[2]
        label_text = ent2text[label]
        head_ent_text = ent2text[line[0]]
        tail_ent_text = ent2text[line[2]]
        relation_text = rel2text[line[1]]
    else:
        ori, line, label = line[0], line[1], line[2] # （triple, cor, label）# label是正确的
        head = 1 if ori[0] == label else 0
        label_text = ent2text[label]
        head_ent_text = ent2text[ori[0]]
        tail_ent_text = ent2text[ori[2]]
        relation_text = rel2text[ori[1]]
    examples = []
    
    entities = list(ent2text.keys())
    
    i = 0

    guid = "%s-%s" % (set_type, i)
    text_a = head_ent_text
    text_b = relation_text
    text_c = tail_ent_text

    if set_type == "dev" or set_type == "test" or set_type == "memory":

        guid = "%s-%s" % (set_type, i)
        examples.append(
            InputExample(
            guid=guid,
            text_a=text_a,
            text_b=text_b,
            text_c=text_c,
            label=1,
            wrong_label=ent2text[line[0]] if head == 1 else ent2text[line[2]],
            target_label=label_text,
            cor_head=head,
            en=[ent2id[line[0]], rel2id[line[1]]],
            rel=rel2id[line[1]]))
        
    elif set_type == "train":
        guid = "%s-%s" % (set_type, i)
        examples.append(
            InputExample(
            guid=guid,
            text_a=text_a,
            text_b=text_b,
            text_c=text_c,
            label=1,
            wrong_label=ent2text[line[0]] if head == 1 else ent2text[line[2]],
            target_label=label_text,
            cor_head=head,
            en=[ent2id[line[0]], rel2id[line[1]]],
            rel=rel2id[line[1]]))

    return examples



def filter_init(head, tail, t1, t2, ent2id_, ent2token_, rel2id_):
    global head_filter_entities
    global tail_filter_entities
    global ent2text
    global rel2text
    global ent2id
    global ent2token
    global rel2id

    head_filter_entities = head
    tail_filter_entities = tail
    ent2text = t1
    rel2text = t2
    ent2id = ent2id_
    ent2token = ent2token_
    rel2id = rel2id_


def delete_init(ent2text_):
    global ent2text
    ent2text = ent2text_


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self, tokenizer, args):
        self.labels = set()
        self.tokenizer = tokenizer
        self.args = args
        self.entity_path = os.path.join(args.data_dir, "entity2text.txt")
        self.task_name = args.task_name

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.get_train_triples(data_dir), "train",
                                     data_dir, self.args)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.get_dev_triples(data_dir), "dev",
                                     data_dir, self.args)

    def get_memory_examples(self, data_dir, chunk=""):
        """See base class."""
        return self._create_examples(self.get_memory_triples(data_dir), "memory",
                                     data_dir, self.args)

    def get_test_examples(self, data_dir, chunk=""):
        """See base class."""
        return self._create_examples(self.get_test_triples(data_dir, chunk),
                                     "test", data_dir, self.args)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip().split('\t')[0])
        rel2token = {ent: f"[RELATION_{i}]" for i, ent in enumerate(relations)}
        return list(rel2token.values())

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        relation = []
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                relation.append(line.strip().split("\t")[-1])
        return relation

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        with open(self.entity_path, 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])

        ent2token = {ent: f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        return list(ent2token.values())

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        if self.args.pretrain:
            return self._read_tsv(os.path.join(
                data_dir,
                "pretrain.tsv"))
        else:
            return self._read_jsonl(os.path.join(
                data_dir,
                "train.jsonl")) if self.task_name == "add" else self._read_jsonl(
                    os.path.join(data_dir, "edit_train.jsonl"))
    
    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_jsonl(os.path.join(
            data_dir,
            "train.jsonl")) if self.task_name == "add" else self._read_jsonl(
                os.path.join(data_dir, "edit_test.jsonl"))

    def get_test_triples(self, data_dir, chunk=""):
        """Gets test triples."""
        return self._read_jsonl(os.path.join(
            data_dir,
            "train.jsonl")) if self.task_name == "add" else self._read_jsonl(
                os.path.join(data_dir, "edit_test.jsonl"))

    def get_memory_triples(self, data_dir):
        """Gets training triples."""
        return self._read_jsonl(os.path.join(data_dir, "stable.jsonl"))

    def _create_examples(self, lines, set_type, data_dir, args):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        ent2text_with_type = {}
        with open(self.entity_path, 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                end = temp[1]  #.find(',')
                if "wiki" in data_dir:
                    assert "Q" in temp[0]
                ent2text[temp[0]] = temp[1].replace("\\n",
                                                    " ").replace("\\",
                                                                 "")  #[:end]

        entities = list(ent2text.keys())
        ent2token = {ent: f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        ent2id = {ent: i for i, ent in enumerate(entities)}

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]
        relation_names = {}
        with open(os.path.join(data_dir, "relations.txt"), "r") as file:
            for line in file.readlines():
                t = line.strip()
                relation_names[t] = rel2text[t]

        tmp_lines = []
        not_in_text = 0
        for line in tqdm(lines, desc="delete entities without text name."):
            res = line
            if isinstance(line, tuple):
                line = line[1]
            if (line[0] not in ent2text) or (line[2] not in ent2text) or (
                    line[1] not in rel2text):
                not_in_text += 1
                continue
            tmp_lines.append(res)
        lines = tmp_lines
        print(f"total entity not in text : {not_in_text} ")

        # rel id -> relation token id
        num_entities = len(self.get_entities(args.data_dir))
        rel2id = {
            w: i + num_entities
            for i, w in enumerate(relation_names.keys())
        }

        # add reverse relation
        # tmp_rel2id = {}
        # num_relations = len(rel2id)
        # cnt = 0
        # for k, v in rel2id.items():
        #     tmp_rel2id[k + " (reverse)"] = num_relations + cnt
        #     cnt += 1
        # rel2id.update(tmp_rel2id)

        examples = []
        # head filter head entity
        head_filter_entities = defaultdict(list)
        tail_filter_entities = defaultdict(list)

        dataset_list = [
            self.get_train_triples(data_dir),
            self.get_dev_triples(data_dir),
            self.get_test_triples(data_dir)
        ]
        # in training, only use the train triples
        if args.pretrain:
            dataset_list = dataset_list[0:1]
        for train_lines in dataset_list:
            # with open(os.path.join(data_dir, m), 'r') as file:
            #     train_lines = file.readlines()
            #     for idx in range(len(train_lines)):
            #         train_lines[idx] = train_lines[idx].strip().split("\t")

            for line in train_lines:
                if isinstance(line, tuple):
                    tail_filter_entities["\t".join([line[1][0],
                                                    line[1][1]])].append(line[1][2])
                    head_filter_entities["\t".join([line[1][2],
                                                    line[1][1]])].append(line[1][0])
                else:
                    tail_filter_entities["\t".join([line[0],
                                                    line[1]])].append(line[2])
                    head_filter_entities["\t".join([line[2],
                                                    line[1]])].append(line[0])

        max_head_entities = max(len(_) for _ in head_filter_entities.values())
        max_tail_entities = max(len(_) for _ in tail_filter_entities.values())

        print(
            f"max number of filter entities : {max_head_entities} {max_tail_entities}"
        )

        from os import cpu_count
        threads = min(1, cpu_count())
        filter_init(head_filter_entities, tail_filter_entities, ent2text,
                    rel2text, ent2id, ent2token, rel2id)

        if hasattr(args, "faiss_init") and args.faiss_init:
            annotate_ = partial(solve_get_knowledge_store,
                                pretrain=self.args.pretrain)
        else:
            annotate_ = partial(solve, pretrain=self.args.pretrain, task_name=self.args.task_name, set_type=set_type)
        examples = list(
            tqdm(map(annotate_, lines),
                 total=len(lines),
                 desc="convert text to examples"))

        tmp_examples = []
        for e in examples:
            for ee in e:
                tmp_examples.append(ee)
        examples = tmp_examples
        # delete vars
        del head_filter_entities, tail_filter_entities, ent2text, rel2text, ent2id, ent2token, rel2id
        return examples


class Verbalizer(object):

    def __init__(self, args):
        if "WN18RR" in args.data_dir:
            self.mode = "WN18RR"
        elif "FB15k" in args.data_dir:
            self.mode = "FB15k"
        elif "umls" in args.data_dir:
            self.mode = "umls"

    def _convert(self, head, relation, tail):
        if self.mode == "umls":
            return f"The {relation} {head} is "

        return f"{head} {relation}"

class KGCDataset(Dataset):

    def __init__(self, features, sampler, config):
        self.features = features
        self.config = config
        # TODO 
        self.sampler = sampler

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)
      
    def edit_generator(self, n_edits=8, n=None, memory=None):
        if n is None:
            n = len(self.features)
        loc_num = len(memory)
        sampler = EditBatchSampler(n, loc_num, memorize_mode=self.config.single_batch, seed=self.config.seed, n_edits=n_edits, stable_batch_size=self.config.stable_batch_size)
        while True:
            edit_idxs, loc_idxs = sampler.sample()
            # assert len(edit_idxs) == 1
            # idxs = loc_idxs + edit_idxs
            edit_feature = [self.features[idx] for idx in edit_idxs]
            edit_batch = next(self.sampler(edit_feature))

            loc_feature = [memory[idx] for idx in loc_idxs]
            loc_batch = next(self.sampler(loc_feature))
            # toks = self.collate_fn([self[idx] for idx in idxs])

            pass_keys = ["input_ids", "attention_mask", "token_type_ids", "label"]
            edit_inner = {k: v for k, v in edit_batch.items() if k in pass_keys} # src的tokenizer，取最后edit_num个edit_idx的
            loc = {k: v for k, v in loc_batch.items() if k in pass_keys} # loc的tokenizer
            cond = None
            # cond = {"input_ids": toks["cond_input_ids"], "attention_mask": toks["cond_attention_mask"]}

            batch = {
                "edit_inner": edit_inner, # src
                "edit_outer": edit_inner, # rephrase
                "loc": loc, # 衡量无关edit的变化程度
                "cond": cond # cond的tokenizer，cond有一半概率用ori一般概率用替换的
            }
            
            
            yield dict_to(batch, self.config.device)



def convert_examples_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(example, max_seq_length, mode, pretrain=1):
    """Loads a data file into a list of `InputBatch`s."""
    # tokens_a = tokenizer.tokenize(example.text_a)
    # tokens_b = tokenizer.tokenize(example.text_b)
    # tokens_c = tokenizer.tokenize(example.text_c)

    # _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length= max_seq_length)
    text_a = " ".join(example.text_a.split()[:128])
    text_b = " ".join(example.text_b.split()[:128])
    text_c = " ".join(example.text_c.split()[:128])

    if pretrain:
        input_text_a = text_a
        input_text_b = text_b
    else:
        input_text_a = tokenizer.sep_token.join([text_a, text_b])
        input_text_b = text_c

    inputs = tokenizer(
        input_text_a,
        input_text_b,
        truncation="longest_first",
        max_length=max_seq_length,
        padding="longest",
        add_special_tokens=True,
    )
    # assert tokenizer.mask_token_id in inputs.input_ids, "mask token must in input"

    features = asdict(
        InputFeatures(input_ids=inputs["input_ids"],
                      attention_mask=inputs['attention_mask'],
                      labels=torch.tensor(example.label),
                      label=torch.tensor(example.real_label)))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()


# @cache_results(_cache_fp="./dataset")
def get_dataset(args, processor, label_list, tokenizer, mode, sampler = None):

    assert mode in ["train", "dev", "test", "memory"], "mode must be in train dev test!"

    # use training data to construct the entity embedding
    # combine_train_and_test = False
    # if args.faiss_init and mode == "test" and not args.pretrain:
    #     mode = "train"
    #     if "ind" in args.data_dir: combine_train_and_test = True
    # else:
    #     pass
    if mode == "memory":
        train_examples = processor.get_memory_examples(args.data_dir)
    elif mode == "train":
        train_examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        train_examples = processor.get_dev_examples(args.data_dir)
    else:
        train_examples = processor.get_test_examples(args.data_dir)


    from os import cpu_count
    with open(os.path.join(args.data_dir, f"examples_{mode}.txt"),
              'w') as file:
        for line in train_examples:
            d = {}
            d.update(line.__dict__)
            file.write(json.dumps(d) + '\n')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=False)

    features = []

    file_inputs = [os.path.join(args.data_dir, f"examples_{mode}.txt")]
    file_outputs = [os.path.join(args.data_dir, f"features_{mode}.txt")]

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin for input in file_inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout for output in file_outputs
        ]

        encoder = MultiprocessingEncoder(tokenizer, args)
        pool = Pool(16, initializer=encoder.initializer)
        encoder.initializer()
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 1000)
        # encoded_lines = map(encoder.encode_lines, zip(*inputs))

        stats = Counter()
        for i, (filt, enc_lines) in tqdm(enumerate(encoded_lines, start=1),
                                         total=len(train_examples)):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    features.append(eval(enc_line))
            else:
                stats["num_filtered_" + filt] += 1

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

    pool.close()
    pool.join()
    for f_id, f in enumerate(features):
        en = features[f_id].pop("en")
        rel = features[f_id].pop("rel")
        cnt = 0
        if not isinstance(en, list): continue

        pos = 0
        for i, t in enumerate(f['input_ids']):
            if t == tokenizer.pad_token_id:
                features[f_id]['input_ids'][i] = en[cnt] + len(tokenizer)
                cnt += 1
            if cnt == len(en): break
        # assert not (args.faiss_init and pos == 0)
        features[f_id]['pos'] = pos

        # for i,t in enumerate(f['input_ids']):
        #     if t == tokenizer.pad_token_id:
        #         features[f_id]['input_ids'][i] = rel + len(tokenizer) + num_entities
        #         break

    features = KGCDataset(features, sampler, args)
    return features


class MultiprocessingEncoder(object):

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.pretrain = args.pretrain
        self.max_seq_length = args.max_seq_length

    def initializer(self):
        global bpe
        bpe = self.tokenizer

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                return ["EMPTY", None]
            # enc_lines.append(" ".join(tokens))
            enc_lines.append(
                json.dumps(
                    self.convert_examples_to_features(example=eval(line))))
            # enc_lines.append(" ")
            # enc_lines.append("123")
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

    def convert_examples_to_features(self, example):
        pretrain = self.pretrain
        max_seq_length = self.max_seq_length
        global bpe
        """Loads a data file into a list of `InputBatch`s."""

        text_a = example['text_a']
        text_b = example['text_b']
        text_c = example['text_c']
        
        inputs = bpe(
            text_a,
            text_b,
            text_c,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="longest",
        )
        # assert bpe.mask_token_id in inputs.input_ids, "mask token must in input"

        features = asdict(
            InputFeatures(input_ids=inputs["input_ids"],
                          attention_mask=inputs['attention_mask'],
                          token_type_ids=inputs['token_type_ids'],
                          label=example['label'],
                          cor_head=example['cor_head'],
                          en=example['en'],
                          rel=example['rel']))
        return features


if __name__ == "__main__":
    dataset = KGCDataset('./dataset')
