from re import DEBUG
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

from models.utils import get_entity_spans_pre_processing


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
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):
            my_args = args[0]
            mode = args[-1]
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True
            
            model_name = my_args.model_name_or_path.split("/")[-1]
            cache_filepath = os.path.join(my_args.data_dir, f"cached_{mode}_features{model_name}.pkl")
            refresh = my_args.overwrite_cache

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
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

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, filter_entities=None, label_with_type=None, relation_ids=None, demos=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label_with_type = label_with_type
        self.label = label
        self.filter_entities = filter_entities
        self.relation_ids = relation_ids
        self.demos = demos


@dataclass
class InputFeatures:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor = None
    filter_ent_ids: torch.Tensor = None
    input_sentences: str = None
    relation_ids: torch.Tensor = None


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

import copy

def binary_search(target , l):
    st = 0
    ed = len(l) - 1
    while st < ed:
        mid = (st+ed) // 2
        if l[mid] > target:
            ed = mid - 1
        elif l[mid] < target:
            st = mid + 1
        else:
            return True
    return False

def solve(line,  set_type="train"):
    examples = []
        
    head_ent_text = ent2text[line[0]]
    tail_ent_text = ent2text[line[2]]
    relation_text = rel2text[line[1]]
    
    i=0
    
    a = tail_filter_entities["\t".join([line[0],line[1]])]
    b = head_filter_entities["\t".join([line[2],line[1]])]
    
   


  
    guid = "%s-%s" % (set_type, i)
    text_a = head_ent_text
    text_b = relation_text
    text_c = tail_ent_text 
    
    # demo_input = demos.get_demo_input(ent2text, rel2text, line[1])
    demo_input = ""

    if line[2] not in ent2text_with_type:
        label_with_type = ""
    else:
        label_with_type = ent2text_with_type[line[2]]
    
    
    if line[2] not in ent2text_with_type:
        label_with_type_2 = ""
    else:
        label_with_type_2 = ent2text_with_type[line[0]]

    examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=text_c, filter_entities=a, relation_ids=rel2id[text_b], demos=demo_input, label_with_type=label_with_type))
    # examples.append(
    #     InputExample(guid=guid, text_a=text_c, text_b=text_b + " (reverse)", text_c = text_a, label=text_a, filter_entities=b, relation_ids=rel2id[text_b+" (reverse)"], demos=demo_input, label_with_type=label_with_type_2))
    return examples

def filter_init(head, tail, t1,t2, t3, rel2id_, t5):
    global head_filter_entities
    global tail_filter_entities
    global ent2text
    global rel2text
    global ent2text_with_type
    global rel2id
    global demos

    head_filter_entities = head
    tail_filter_entities = tail
    ent2text =t1
    rel2text =t2
    ent2text_with_type = t3
    rel2id = rel2id_
    demos = t5

def delete_init(ent2text_):
    global ent2text
    
    ent2text = ent2text_


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir, chunk=""):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, f"test{chunk}.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

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
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir, chunk=""):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, f"test{chunk}.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        ent2text_with_type = {}
        # with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
        #     ent_lines = f.readlines()
        #     for line in ent_lines:
        #         temp = line.strip().split('\t')
        #         end = temp[1]#.find(',')
        #         if "wiki" in data_dir:
        #             assert "Q" in temp[0]
        #         ent2text[temp[0]] = temp[1] #[:end]
  
        ent2text_with_type = {}
        ent2id = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            cnt = 0
            for line in ent_lines:
                temp = line.strip().split('\t')
                #first_sent_end_position = temp[1].find(".")
                ent2id[temp[0]] = cnt
                cnt += 1
                ent2text[temp[0]] = temp[1]#[:first_sent_end_position + 1] 

            # with open(os.path.join(data_dir, "entity2text_withtype.txt"), 'r') as f:
            #     ent_lines = f.readlines()
            #     cnt = 0
            #     for line in ent_lines:
            #         temp = line.strip().split('\t')
            #         #first_sent_end_position = temp[1].find(".")
            #         cnt += 1
            #         if len(temp) == 1:
            #             import IPython; IPython.embed(); exit(1)
            #         ent2text_with_type[temp[0]] = temp[1]#[:first_sent_end_position + 1] 

        entities = list(ent2text.keys())
        
        # threads = 16
        # with Pool(threads, initializer=delete_init, initargs=(ent2text,)) as pool:
        #     annotate_ = partial(
        #         solve,
        #     )
        #     examples = list(
        #         tqdm(
        #             pool.imap(annotate_, lines, chunksize= 128),
        #             total=len(lines),
        #             desc="convert text to examples"
        #         )
        #     )
        # clean entity
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]      
        relation_names = []
        with open(os.path.join(data_dir, "relations.txt"), "r") as file:
            for line in file.readlines():
                t = line.strip()
                relation_names.append(rel2text[t])

        tmp_lines = []
        not_in_text = 0
        for line in tqdm(lines, desc="delete entities without text name."):
            if (line[0] not in ent2text) or (line[2] not in ent2text) or (line[1] not in rel2text):
                not_in_text += 1
                continue
            tmp_lines.append(line)
        lines = tmp_lines
        print(f"total entity not in text : {not_in_text} ")

        rel2id = {w:i for i,w in enumerate(relation_names)}

        # add reverse relation 
        tmp_rel2id = {}
        num_relations = len(rel2id)
        cnt = 0
        for k, v in rel2id.items():
            tmp_rel2id[k + " (reverse)"] = num_relations + cnt
            cnt += 1
        rel2id.update(tmp_rel2id)

        examples = []
        # head filter head entity
        head_filter_entities = defaultdict(list)
        tail_filter_entities = defaultdict(list)

        # set filter
        with open(os.path.join(data_dir, "train.tsv"), 'r') as file:
            train_lines = file.readlines()
            for idx in range(len(train_lines)):
                train_lines[idx] = train_lines[idx].strip().split("\t")
            for line in train_lines:
                tail_filter_entities["\t".join([line[0], line[1]])].append(ent2id[line[2]])
                head_filter_entities["\t".join([line[2], line[1]])].append(ent2id[line[0]])

        with open(os.path.join(data_dir, "dev.tsv"), 'r') as file:
            train_lines = file.readlines()
            for idx in range(len(train_lines)):
                train_lines[idx] = train_lines[idx].strip().split("\t")
            for line in train_lines:
                tail_filter_entities["\t".join([line[0], line[1]])].append(ent2id[line[2]])
                head_filter_entities["\t".join([line[2], line[1]])].append(ent2id[line[0]])


        with open(os.path.join(data_dir, "test.tsv"), 'r') as file:
            train_lines = file.readlines()
            for idx in range(len(train_lines)):
                train_lines[idx] = train_lines[idx].strip().split("\t")
            for line in train_lines:
                tail_filter_entities["\t".join([line[0], line[1]])].append(ent2id[line[2]])
                head_filter_entities["\t".join([line[2], line[1]])].append(ent2id[line[0]])


        
        
        max_head_entities = max(len(_) for _ in head_filter_entities.values())
        max_tail_entities = max(len(_) for _ in tail_filter_entities.values())
        
        print(f"max number of filter entities : {max_head_entities} {max_tail_entities}")

        
        from os import cpu_count
        threads = min(1, cpu_count())
        # demos = Demos(data_dir)
        demos = ""
        filter_init(head_filter_entities, tail_filter_entities,ent2text, rel2text,
            ent2text_with_type, rel2id,demos ,)
        
        annotate_ = partial(
                solve,
            )
        examples = list(
            tqdm(
                map(annotate_, lines),
                total=len(lines),
                desc="convert text to examples"
            )
        )


        # with Pool(threads, initializer=filter_init, initargs=(head_filter_entities, tail_filter_entities,ent2text, rel2text,
        #     ent2text_with_type, rel2id,)) as pool:
        #     annotate_ = partial(
        #         solve,
        #     )
        #     examples = list(
        #         tqdm(
        #             map(annotate_, lines, chunksize= 128),
        #             total=len(lines),
        #             desc="convert text to examples"
        #         )
        #     )
        tmp_examples = []
        for e in examples:
            for ee in e:
                tmp_examples.append(ee)
        examples = tmp_examples
        # delete vars
        del head_filter_entities, tail_filter_entities, ent2text, rel2text, ent2text_with_type, rel2id, demos
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

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
class PretrainKGCDataset(IterableDataset):
    def __init__(self, args, mode) -> None:
        super().__init__()
        self.args = args
        self.file_name = os.path.join(args.data_dir, f"features_{mode}.txt")
        pass
        
    def parse_file(self):
        with open(self.file_name, 'r') as file:
            idx = 0
            lines = []
            for line in file.readlines():
                idx += 1
                lines.append(line)
                if idx % 2 == 0:
                    yield self._convert_lines_to_feature(lines)
                    lines = []
    
    def _convert_lines_to_feature(self, lines):
        line1, line2 = lines
        input_ids = torch.tensor(list(map(int,line1.strip().split())))[:self.args.max_seq_length]
        labels = torch.tensor(list(map(int,line2.strip().split())))
        attention_mask = torch.ones_like(input_ids)
        feature = asdict(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=labels))
        return feature
    
    def __iter__(self):
        return self.parse_file()
    
    def __len__(self):
        return len(open(self.file_name, 'r').readlines()) // 2


    

def convert_examples_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def convert_examples_to_features(example, max_seq_length, model_name_or_path, mode, output_full_sentence=True, entity_token="<entity>", prefix_tree_decode=0, use_label_type=0, use_demos=0):
    """Loads a data file into a list of `InputBatch`s."""
    # max_seq_length = args.max_seq_length



    # tokens_a = tokenizer.tokenize(example.text_a)
    # tokens_c = tokenizer.tokenize(example.text_c)

    # _truncate_seq_pair(tokens_a, tokens_c, max_seq_length)

    # example.text_a = tokenizer.convert_tokens_to_string(tokens_a)
    # example.text_c = tokenizer.convert_tokens_to_string(tokens_c)

    assert example.label == example.text_a or example.label == example.text_c

    """
    """
    # verbalizer = Verbalizer(args)

    # 43.9 origin score
    text = f"{example.text_a} {example.text_b} "
    hide = example.text_a if example.label == example.text_c else example.text_c
    # if not output_full_sentence:
    #     if example.text_a == example.label:
    #         text = f" {example.text_c} {example.text_b}"
    #     else:
    #         text = f" {example.text_a} {example.text_b}"
    #     input_text = text
    # else:
    #     mask_token = tokenizer.mask_token if not prefix_tree_decode else entity_token
    #     text = f"{example.text_a} {example.text_b} {example.text_c} .".strip()
    #     # text = f"{example.text_a} {tokenizer.sep_token}  {example.text_b} {tokenizer.sep_token} {example.text_c} .".strip()
    #     input_text = text.replace(example.label, mask_token)
    # t_pattern = pattern[example.text_b.strip()]
    # text = f"{t_pattern[0]} {example.text_a} {t_pattern[1]} {example.text_b} {t_pattern[2]} {example.text_c} {t_pattern[3]}"
    # text = get_entity_spans_pre_processing([text])[0]
    # t5 text filling task
    # if "t5" in model_name_or_path:
    #     text = f"A {example.text_a} is a {example.text_b} of {example.text_c} ."

    input_text = text
    
    # add demos 
    # input_text = example.demos + input_text if use_demos else input_text
    # input_text ++

    # input_text += f"[RELATION_{example.relation_ids}]"
    
    random.shuffle(example.filter_entities)
    mask_token = tokenizer.mask_token if "T5" not in tokenizer.__class__.__name__ else "<extra_id_0>"
    # mask_token = " "
        
    # add head and tail entity
    features = []
    inputs = tokenizer(
        input_text,
        truncation="longest_first",
        max_length=max_seq_length,
        padding="longest",
        add_special_tokens=True,
    )

    ent = example.filter_entities
    # ent = tokenizer(
    #     filter_entities,
    #     truncation="longest_first",
    #     max_length=max_seq_length,
    #     padding=False,
    #     add_special_tokens=False,
    # ).input_ids

    if mode == "train":
        if output_full_sentence:
            label_text = text
        else:
            label_text = example.label
    else:
        label_text = example.label
    # change the origin text to  `head relation { <entity> } [ tail ]`
    if prefix_tree_decode and output_full_sentence:
        label_text = text.replace(example.label, " { <entity> } [ " + example.label + " ] ")
        # text = get_entity_spans_pre_processing([text])[0]
    if mode != "train":
        label_text = example.label

    if use_label_type:
        label_text = example.label_with_type
    # if output_full_sentence and mode == "train" and nOt prefix_tree_decode:
    #     label_text = text

    # else:
    #     label_text = example.label.strip()

    #  target sentence  head is a relation of {<entity>} [ entity ]
    # label_text = label_text.replace(example.label, " { " + entity_token + " } [ " + example.label + " ] ")
    # label_text = example.label_with_type

    # label_text += f"[RELATION_{example.relation_ids}]"

    labels = tokenizer(
        label_text,
        truncation="longest_first",
        max_length=max_seq_length,
        padding="longest",
        add_special_tokens=True,
    ).input_ids

    # if random.random() <= 0.001:
    #     print("input text:  " + text.replace(example.label, tokenizer.mask_token))
    #     print("label:  " + example.label)
    #     print("filter entity:  " + " \t ".join(filter_entities))
        # if "t5" in args.model_name_or_path:
        #     labels = [0] + labels

    if mode != "train":
        features = asdict(
            InputFeatures(input_ids=inputs["input_ids"],
                            attention_mask=inputs['attention_mask'],
                            labels=labels,
                            filter_ent_ids = ent,
                            # input_sentences=text.replace(example.label, entity_token),
                            relation_ids=torch.tensor(example.relation_ids)
            )
        )
    else:
        features = asdict(InputFeatures(input_ids=inputs["input_ids"],
                                attention_mask=inputs['attention_mask'],
                                labels=labels,
                                filter_ent_ids = ent,
                                relation_ids=torch.tensor(example.relation_ids)
            )
        )
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

# 通过relation ids返回entity ids, 需要有ent2text
class Demos(object):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # self._get_relation()
        # self._get_demos()
    
    def _get_relation(self):
        self.rel2demos = {}
        with open(os.path.join(self.data_dir, "relation2text.txt"), "r") as file:
            for line in file.readlines():
                rel = line.strip().split("\t")[0]
                self.rel2demos[rel] = []
    
    def _get_demos(self):
        with open(os.path.join(self.data_dir, "train.tsv"), "r") as file:
            for line in file.readlines():
                h, r, t = line.strip().split("\t")
                self.rel2demos[r].append([h, t])
        
        for k, v in self.rel2demos.items():
            random.shuffle(self.rel2demos[k])
            self.rel2demos[k] = self.rel2demos[k][:2]


    def get_demo_input(self, ent2text, rel2text, relation):
        triples = self.rel2demos[relation]
        input_text = " "
        for h, t in triples:
            input_text += " ".join([ent2text[h], rel2text[relation], ent2text[t]]) + " <sep>"
            input_text += " ".join([ent2text[t], rel2text[relation] + " (reverse)", ent2text[h]]) + " <sep>"
        
        
        return input_text

# @cache_results(_cache_fp="./dataset")
def get_dataset(args, processor, label_list, tokenizer, mode):
    
    if mode == "train" and ("YAGO" in args.data_dir or "Ali" in args.data_dir):
        return PretrainKGCDataset(args, mode)
        # features = []
        # name = mode + "_0" if mode == "train" else mode
        # with open(os.path.join(args.data_dir, f"features_{name}.txt"), "r") as file:
        #     lines = file.readlines()
        #     len_lines = len(lines)
        #     for i in tqdm(range(len_lines//2)):
        #         line1 = lines[i*2]
        #         line2 = lines[i*2+1]
        #         input_ids = torch.tensor(list(map(int,line1.strip().split())))[:args.max_seq_length]
        #         input_ids[-1] = 2
        #         labels = torch.tensor(list(map(int,line2.strip().split())))[:args.max_seq_length-1]
        #         labels[-1] = 2
        #         attention_mask = torch.ones_like(input_ids)
        #         features.append(asdict(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=labels)))

        # return KGCDataset(features)


    assert mode in ["train", "dev", "test"], "mode must be in train dev test!"

    if mode == "train":
        train_examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        train_examples = processor.get_dev_examples(args.data_dir)
    else:
        train_examples = processor.get_test_examples(args.data_dir)

    from os import cpu_count
    threads = min(16, cpu_count())

    features = []
    # with open(os.path.join(args.data_dir, "cached_relation_pattern.pkl"), "rb") as file:
    #     pattern = pickle.load(file)
    pattern = None
    convert_examples_to_features_init(tokenizer)
    annotate_ = partial(
        convert_examples_to_features,
        max_seq_length=args.max_seq_length,
        model_name_or_path=args.model_name_or_path,
        mode = mode,
        output_full_sentence=args.output_full_sentence,
        prefix_tree_decode = args.prefix_tree_decode,
        use_label_type=args.use_label_type,
        use_demos=args.use_demos
    )
    features = list(
        tqdm(
            map(annotate_, train_examples),
            total=len(train_examples)
        )
    )
    # with Pool(threads, initializer=convert_examples_to_features_init, initargs=(tokenizer,)) as pool:
    #     annotate_ = partial(
    #         convert_examples_to_features,
    #         max_seq_length=args.max_seq_length,
    #         model_name_or_path=args.model_name_or_path,
    #         mode = mode,
    #         output_full_sentence=args.output_full_sentence
    #     )
    #     features = list(
    #         tqdm(
    #             pool.imap(annotate_, train_examples),
    #             total=len(train_examples),
    #             desc="convert examples to features",
    #         )
    #     )
        

    features = KGCDataset(features)
    return features

if __name__ == "__main__":
    dataset = KGCDataset('./dataset')