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
from tqdm import tqdm
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 text_c=None,
                 label=None,
                 real_label=None,
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
        self.real_label = real_label
        self.en = en
        self.rel = rel  # rel id

@dataclass
class InputFeaturesForKE:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    cond_input_ids: torch.Tensor
    cond_attention_mask: torch.Tensor
    labels: torch.Tensor = None
    label: torch.Tensor = None
    en: torch.Tensor = 0
    rel: torch.Tensor = 0
    pos: torch.Tensor = 0

@dataclass
class InputFeaturesForCaliNet:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor = None
    label: torch.Tensor = None
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
        if input_file.split('/')[-1] == 'stable.jsonl' or input_file.split('/')[-1] == 'stable_.jsonl':
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
                    
            return lines

import copy

def solve_kgc(line, data_dir = None, set_type=None, task_name=None):
    if isinstance(line, list):
        line = line
    elif set_type == 'memory':
        _, line, label = line[0], line[1], line[2]
    elif task_name == 'add':
        label, line, head = line[0], line[1], line[2]
    else:
        ori, line, label = line[0], line[1], line[2] # （triple, cor, label）label is correct
        head = 1 if ori[0] == label else 0
    examples = []

    head_ent_text = ent2text[line[0]]
    tail_ent_text = ent2text[line[2]]
    relation_text = rel2text[line[1]]

    i = 0
    
    if set_type == 'memory':
        head = 1 if line[0] == label else 0
        a = [line[0] if head else line[2]]
        label = a[0]
    else:
        if task_name == 'add':
            a = [line[0] if head else line[2]]
        else:
            a = [line[0] if head else line[2]] # a 代表错误的实体

    guid = "%s-%s" % (set_type, i)
    text_a = head_ent_text
    text_b = relation_text
    text_c = tail_ent_text

    
    if head == 1:
        examples.append(
            InputExample(guid=guid,
                        text_a="[MASK]",
                        text_b=text_b + "[PAD]",
                        text_c="[PAD]" + " " + text_c,
                        label=lmap(lambda x: ent2id[x], a), # label= label
                        real_label=ent2id[label],
                        en=[rel2id[line[1]], ent2id[line[2]]],
                        rel=rel2id[line[1]]))
    else :
        examples.append(
            InputExample(guid=guid,
                        text_a="[PAD] ",
                        text_b=text_b + "[PAD]",
                        text_c="[MASK]" + " " + text_a,
                        label=lmap(lambda x: ent2id[x], a),
                        real_label=ent2id[label],
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
        self.entity_path = os.path.join(args.data_dir, "entity2textlong.txt") if os.path.exists(os.path.join(args.data_dir, 'entity2textlong.txt')) \
        else os.path.join(args.data_dir, "entity2text.txt")
        self.task_name = args.task_name

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.get_train_triples(data_dir), "train",
                                     data_dir, self.args)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.get_dev_triples(data_dir), "dev",
                                     data_dir, self.args)

    def get_test_examples(self, data_dir, chunk=""):
        """See base class."""
        return self._create_examples(self.get_test_triples(data_dir, chunk),
                                     "test", data_dir, self.args)

    def get_memory_examples(self, data_dir, chunk=""):
        """See base class."""
        return self._create_examples(self.get_memory_triples(data_dir), "memory",
                                     data_dir, self.args)

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
        return self._read_jsonl(os.path.join(data_dir, "train.jsonl"))

    def get_memory_triples(self, data_dir):
        """Gets training triples."""
        return self._read_jsonl(os.path.join(data_dir, "stable.jsonl"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_jsonl(os.path.join(data_dir, "dev.jsonl"))

    def get_test_triples(self, data_dir, chunk=""):
        """Gets test triples."""
        return self._read_jsonl(os.path.join(data_dir, "test.jsonl"))

    def _create_examples(self, lines, set_type, data_dir, args):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
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

        examples = []
        # head filter head entity
        head_filter_entities = defaultdict(list)
        tail_filter_entities = defaultdict(list)


        from os import cpu_count
        threads = min(1, cpu_count())
        filter_init(head_filter_entities, tail_filter_entities, ent2text,
                    rel2text, ent2id, ent2token, rel2id)
        
        annotate_ = partial(solve_kgc, task_name=self.args.task_name, set_type=set_type)
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

    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)

def get_dataset(args, processor, label_list, tokenizer, mode):

    assert mode in ["train", "dev", "test", "memory"], "mode must be in train dev test or memory!"
    
    add_tokenizer = tokenizer
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

    encoders = {
      'KGEditor': KEMultiprocessingEncoder,
      'KE': KEMultiprocessingEncoder,
      'CaliNet': CaliNetMultiprocessingEncoder,
      'K-Adapter': CaliNetMultiprocessingEncoder,
    }
    
    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin for input in file_inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout for output in file_outputs
        ]

        encoder = encoders[args.kge_model_type](add_tokenizer, args, tokenizer)
        pool = Pool(16, initializer=encoder.initializer)
        encoder.initializer()
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 1000)

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
        real_label = f['label']
        cnt = 0
        if not isinstance(en, list): continue

        pos = 0
        for i, t in enumerate(f['input_ids']):
            if t == tokenizer.pad_token_id:
                features[f_id]['input_ids'][i] = en[cnt] + len(tokenizer)
                cnt += 1
            if features[f_id]['input_ids'][i] == real_label + len(tokenizer):
                pos = i
            if cnt == len(en): break
            
        features[f_id]['pos'] = pos

    features = KGCDataset(features)
    return features
  
class KEMultiprocessingEncoder(object):

    def __init__(self, add_tokenizer, args, tokenizer):
        self.tokenizer = tokenizer
        self.add_tokenizer = add_tokenizer
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
            enc_lines.append(
                json.dumps(
                    self.convert_examples_to_features(example=eval(line))))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

    def convert_examples_to_features(self, example):
        max_seq_length = self.max_seq_length
        global bpe
        """Loads a data file into a list of `InputBatch`s."""

        text_a = example['text_a']
        text_b = example['text_b']
        text_c = example['text_c']
            
        if text_a == "[MASK]":
            input_text_a = bpe.sep_token.join([text_a, text_b])
            input_text_b = text_c
        else:
            input_text_a = text_a
            input_text_b = bpe.sep_token.join([text_b, text_c])

        cond_inputs = "{} >> {} || {}".format(
            self.add_tokenizer.added_tokens_decoder[example['label'][0] + len(self.tokenizer)],
            self.add_tokenizer.added_tokens_decoder[example['real_label'] + len(self.tokenizer)],
            input_text_a + input_text_b,
        )

        inputs = bpe(
            input_text_a,
            input_text_b,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="longest",
            add_special_tokens=True,
        )

        cond_inputs = bpe(
            cond_inputs,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            add_special_tokens=True,
        )
                        
        features = asdict(
            InputFeaturesForKE(input_ids=inputs["input_ids"],
                          attention_mask=inputs['attention_mask'],
                          cond_input_ids=cond_inputs["input_ids"],
                          cond_attention_mask=cond_inputs["attention_mask"],
                          labels=example['label'],
                          label=example['real_label'],
                          en=example['en'],
                          rel=example['rel']))
        return features
      
class CaliNetMultiprocessingEncoder(KEMultiprocessingEncoder):

    def __init__(self, add_tokenizer, args, tokenizer):
        super().__init__(add_tokenizer, args, tokenizer)

    def convert_examples_to_features(self, example):
        max_seq_length = self.max_seq_length
        global bpe
        """Loads a data file into a list of `InputBatch`s."""

        text_a = example['text_a']
        text_b = example['text_b']
        text_c = example['text_c']
            
        if text_a == "[MASK]":
            input_text_a = bpe.sep_token.join([text_a, text_b])
            input_text_b = text_c
        else:
            input_text_a = text_a
            input_text_b = bpe.sep_token.join([text_b, text_c])

        inputs = bpe(
            input_text_a,
            input_text_b,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="longest",
            add_special_tokens=True,
        )
        
        assert bpe.mask_token_id in inputs.input_ids, "mask token must in input"
                 
        features = asdict(
            InputFeaturesForCaliNet(input_ids=inputs["input_ids"],
                          attention_mask=inputs['attention_mask'],
                          labels=example['label'],
                          label=example['real_label'],
                          en=example['en'],
                          rel=example['rel']))
        return features
  
process_mapping = {
    'KGEditor': KGProcessor,
    'KE': KGProcessor,
    'CaliNet': KGProcessor,
    'K-Adapter': KGProcessor,
}