from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import os
import random
import json
import torch
import copy
import numpy as np
import pickle
from tqdm import tqdm



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

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = _pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    _prepare_cache_filepath(cache_filepath)
                    with open(cache_filepath, 'wb') as f:
                        _pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_

class KGCDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super().__init__()
        self.args = args
        self.dataset = self.args.dataset
        self.mode = mode
        self.train_tasks = json.load(open(self.dataset + f'/{mode}_tasks.json'))
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        self.task_pool = list(self.train_tasks.keys())
        self.tokenizer = tokenizer

        # ignore the candidates less or equal than 20
        # if mode == "train":
        #     self._select_samples()
        num_tasks = len(self.task_pool)


        self._load_symbol2id()
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        self._gather_samples()

    def _select_samples(self):
        temp_task_pool = []
        for i, rel in enumerate(self.task_pool):
            if len(self.rel2candidates[rel]) > 20:
                temp_task_pool.append(rel)

        self.task_pool = temp_task_pool


    def test(self, index):
        #  one batch for every rel
        few = self.args.few
        # rel, type=str
        task_choice = self.task_pool[index]
        candidates = self.rel2candidates[task_choice]

        symbol2id = self.symbol2id
        ent2id = self.ent2id

        task_triples = self.train_tasks[task_choice]
        random.shuffle(task_triples)

        # pick the few triples as positive and the rest for the negative
        support_triples = task_triples[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]



        # 
        other_triples = task_triples[few:]
        assert len(other_triples), "I hope it will not happen"
        if self.mode == "train":
            if len(other_triples) < self.args.batch_size:
                # 重采样了
                query_triples = [random.choice(other_triples) for _ in range(self.args.batch_size)]
            else:
                query_triples = random.sample(other_triples, self.args.batch_size)
        else:
            # select the other triples as the query set
            query_triples = other_triples

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]
        
        # get the candidates tokens

        false_pairs = []
        false_left = []
        false_right = []
        for triple in query_triples:
            e_h = triple[0]
            rel = triple[1]
            e_t = triple[2]
            while True:
                noise = random.choice(candidates)  # select noise from candidates
                if (noise not in self.e1rel_e2[e_h + rel]) \
                        and noise != e_t:
                    break
            false_pairs.append([symbol2id[e_h], symbol2id[noise]])
            false_left.append(ent2id[e_h])
            false_right.append(ent2id[noise])
        
        # We Union!
        # use hinge loss,  dame
        # build the text
        reference_text = "The reference is that "
        for triple in support_triples:
            reference_text += self._ju(triple) + " , "
        
        
        query_text = []
        labels = []
        for triple in query_triples:
            query_text.append(self._ju(triplet=triple[:2]))
            query_text[-1] += " ".join([self.tokenizer.mask_token]*self.args.num_entity_tokens)  + "."
            temp_labels = self.tokenizer(
                self._triplet2text(triple[2]),
                add_special_tokens=False,
            )['input_ids']
            if len(temp_labels) <= self.args.num_entity_tokens:
                temp_labels = temp_labels + [-100] * (self.args.num_entity_tokens - len(temp_labels))
            else:
                while len(temp_labels) > self.args.num_entity_tokens:
                    temp_labels.pop()
            assert len(temp_labels) == self.args.num_entity_tokens, "pop or add!"

            labels.append(temp_labels)

        if self.mode != "train":
            test_labels = copy.deepcopy(labels)
            test_labels = torch.tensor(test_labels)

        # modify the labels, set the entity instead of the other things

        candidates_ids = self.tokenizer(
            [self._triplet2text(candidate) for candidate in candidates],
            add_special_tokens=False,
            truncation="longest_first",
            max_length=self.args.num_entity_tokens,
            padding="max_length",
            return_tensors='pt'
        )['input_ids']

        text = [reference_text + self.tokenizer.sep_token + "We can infer that " + q for q in query_text]
        inputs = self.tokenizer(
            text,
            truncation="longest_first",
            max_length=self.args.max_seq_length,
            padding="max_length",
            add_special_tokens=True,
            return_tensors='pt'
        )
        #! the first mask token idx, maybe some bugs
        mask_idx = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_select_idx = [i*self.args.num_entity_tokens for i in range(self.args.batch_size)]
        mask_idx = mask_idx[mask_select_idx]




        labels = [[-100]*t + labels[_] + [-100]*(self.args.max_seq_length-t-self.args.num_entity_tokens) for _, t in enumerate(mask_idx)]

        labels = torch.tensor(labels).long()

        input_ids = inputs['input_ids']
        attention_mask = inputs['input_ids']
        token_type_ids = inputs['token_type_ids'] if "token_type_ids" in inputs else None

        if self.mode == "train":
            if token_type_ids is not None:
                return input_ids, attention_mask, token_type_ids, labels
            else:
                return input_ids, attention_mask, labels
        else:
            if token_type_ids is not None:
                return input_ids, attention_mask, token_type_ids, test_labels, candidates_ids
            else:
                return input_ids, attention_mask, test_labels, candidates_ids


    def __getitem__(self, index):
        return self.samples[index]

    def _gather_samples(self):

        pickle_file_name = f"./dataset/NELL-{self.mode}.pkl"
        if os.path.exists(pickle_file_name) and not self.args.overwrite_cache:
            with open(pickle_file_name, "rb") as file:
                self.samples = pickle.load(file)
            return

        self.samples = []
        for index in range(len(self.task_pool)):
            few = self.args.few
            # rel, type=str
            task_choice = self.task_pool[index]
            #! for test
            candidates = self.rel2candidates[task_choice]

            symbol2id = self.symbol2id
            ent2id = self.ent2id

            task_triples = self.train_tasks[task_choice]
            random.shuffle(task_triples)

            # pick the few triples as positive and the rest for the negative
            support_triples = task_triples[:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
            support_left = [ent2id[triple[0]] for triple in support_triples]
            support_right = [ent2id[triple[2]] for triple in support_triples]



            # 
            other_triples = task_triples[few:]
            assert len(other_triples), "I hope it will not happen"
            # if self.mode == "train":
            #     if len(other_triples) < self.args.batch_size:
            #         # 重采样了
            #         query_triples = [random.choice(other_triples) for _ in range(self.args.batch_size)]
            #     else:
            #         query_triples = random.sample(other_triples, self.args.batch_size)
            # else:
                # select the other triples as the query set
            
            # select all the triples
            query_triples = other_triples

            query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
            query_left = [ent2id[triple[0]] for triple in query_triples]
            query_right = [ent2id[triple[2]] for triple in query_triples]
            
            # get the candidates tokens

            false_pairs = []
            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)  # select noise from candidates
                    if (noise not in self.e1rel_e2[e_h + rel]) \
                            and noise != e_t:
                        break
                false_pairs.append(triple[:2] + [noise])

            
            # We Union!
            # use hinge loss,  dame
            # build the text
            reference_text = "The reference is that "
            for triple in support_triples:
                reference_text += self._ju(triple) + " , "
            
            
            query_text = []
            labels = []
            for triple in query_triples:
                query_text.append(self._ju(triplet=triple[:2]))
                query_text[-1] += " ".join([self.tokenizer.mask_token]*self.args.num_entity_tokens)  + "."
                temp_labels = self.tokenizer(
                    self._triplet2text(triple[2]),
                    add_special_tokens=False,
                )['input_ids']
                if len(temp_labels) <= self.args.num_entity_tokens:
                    temp_labels = temp_labels + [-100] * (self.args.num_entity_tokens - len(temp_labels))
                else:
                    while len(temp_labels) > self.args.num_entity_tokens:
                        temp_labels.pop()
                assert len(temp_labels) == self.args.num_entity_tokens, "pop or add!"

                labels.append(temp_labels)

            if self.mode != "train":
                test_labels = copy.deepcopy(labels)
                test_labels = torch.tensor(test_labels)

            # modify the labels, set the entity instead of the other things

            candidates_ids = self.tokenizer(
                [self._triplet2text(candidate) for candidate in candidates],
                add_special_tokens=False,
                truncation="longest_first",
                max_length=self.args.num_entity_tokens,
                padding="max_length",
                return_tensors='pt'
            )['input_ids']

            text = [reference_text + self.tokenizer.sep_token + "We can infer that " + q for q in query_text]
            inputs = self.tokenizer(
                text,
                truncation="longest_first",
                max_length=self.args.max_seq_length,
                padding="max_length",
                add_special_tokens=True,
                return_tensors='pt'
            )
            #! the first mask token idx, maybe some bugs
            mask_idx = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            mask_select_idx = [i*self.args.num_entity_tokens for i in range(self.args.batch_size)]
            mask_idx = mask_idx[mask_select_idx]



            input_ids = inputs['input_ids']

            # tokenize the eval inputs
            ref_entity_idx = set()
            for triple in support_triples:
                t = self._get_entity_tokenize_idx(input_ids[0], self._triplet2text(triple[2]))
                for tt in t:
                    ref_entity_idx.add(tt.item())
            #? hope no bugs
            ref_entity_idx = torch.tensor(list(ref_entity_idx)[:5])
            candidates = np.array(candidates)


            if self.mode == "train":
                for idx, triple in enumerate(query_triples):
                    text = reference_text + self.tokenizer.sep_token + self._ju(triple[:2]) + "@placeholder"
                    # positive pairs
                    p = self.tokenizer(
                        text.replace("@placeholder", self._triplet2text(triple[2])),
                        truncation="longest_first",
                        max_length=self.args.max_seq_length,
                        padding="max_length",
                        add_special_tokens=True,
                        return_tensors='pt'
                    )
                    iidd = self._get_entity_tokenize_idx(p['input_ids'], self._triplet2text(triple[2]))
                    p['entity_idx'] = iidd[-1] 
                    self.samples.append(dict(input_ids=p['input_ids'].squeeze(0), attention_mask=p['attention_mask'].squeeze(0),
                        labels=torch.tensor(1, dtype=torch.int32),
                        entity_idx=p['entity_idx'],
                        ref_entity_idx=ref_entity_idx
                    ))
                    # negative pairs
                    p = self.tokenizer(
                        text.replace("@placeholder", self._triplet2text(false_pairs[idx][2])),
                        truncation="longest_first",
                        max_length=self.args.max_seq_length,
                        padding="max_length",
                        add_special_tokens=True,
                        return_tensors='pt'
                    )
                    iidd = self._get_entity_tokenize_idx(p['input_ids'], self._triplet2text(false_pairs[idx][2]))
                    p['entity_idx'] = iidd[-1] 
                    self.samples.append(dict(input_ids=p['input_ids'].squeeze(0), attention_mask=p['attention_mask'].squeeze(0),
                        labels=torch.tensor(0, dtype=torch.int32),
                        entity_idx=p['entity_idx'],
                        ref_entity_idx=ref_entity_idx
                    ))
            else:
                for triple in tqdm(query_triples):
                    t_candidate = []
                    for idx, ent in enumerate(candidates):
                        # ignore the 1-n and the same entity
                        if ent not in self.e1rel_e2[triple[0]+triple[1]] and ent != triple[2]:
                            t_candidate.append(idx)
                    text = reference_text + self.tokenizer.sep_token + self._ju(triple[:2]) + "@placeholder"
                    # get the positive sample
                    p = self.tokenizer(
                        text.replace("@placeholder", self._triplet2text(triple[2])),
                        truncation="longest_first",
                        max_length=self.args.max_seq_length,
                        padding="max_length",
                        add_special_tokens=True,
                        return_tensors='pt'
                    )
                    iidd = self._get_entity_tokenize_idx(p['input_ids'], self._triplet2text(triple[2]))
                    p['entity_idx'] = iidd[-1] 
                    t = []
                    # get the candidate sample
                    for ent in candidates[t_candidate]:
                        entity_text = self._triplet2text(ent)
                        n = {}
                        # n = self.tokenizer(
                        #     text.replace("@placeholder", entity_text),
                        #     truncation="longest_first",
                        #     max_length=self.args.max_seq_length,
                        #     padding="max_length",
                        #     add_special_tokens=True,
                        #     return_tensors='pt'
                        # )
                        # iidd = self._get_entity_tokenize_idx(n['input_ids'], entity_text)
                        # n['entity_idx'] = iidd[-1] 
                        ttt = self.tokenizer(
                            entity_text + " .",
                            truncation="longest_first",
                            max_length=self.args.num_entity_tokens + 2, # for cls and sep
                            padding="max_length",
                            add_special_tokens=True,
                            return_tensors='pt'
                        )
                        n['replace_entity'] = ttt
                        t.append(n)
                    """
                        replace the input_ids during the val period, due to the limit of memory.
                    
                    """
                    input_ids = p['input_ids']
                    attention_mask = p['attention_mask']
                    entity_idx = p['entity_idx']
                    modified_entity_ids = torch.stack([tt['replace_entity']['input_ids'][0] for tt in t])
                    modified_attention_mask = torch.stack([tt['replace_entity']['attention_mask'][0] for tt in t])

                    self.samples.append(dict(input_ids=input_ids, attention_mask=attention_mask, entity_idx=entity_idx, ref_entity_idx=ref_entity_idx, modify_entity_ids=modified_entity_ids, modify_attention_mask=modified_attention_mask))

        with open(pickle_file_name, "wb") as file:
            pickle.dump(self.samples, file)
        
        print(f'the len of samples is {len(self.samples)} !')

    def _triplet2text(self, text):
        return text.split(":")[-1].replace("_"," ")
    
    def _ju(self, triplet):
        triplet = [self._triplet2text(t) for t in triplet]
        if len(triplet) == 3:
            return f'{triplet[2]} is the {triplet[1]} of {triplet[0]}'
        else:
            return f'{triplet[0]} is the {triplet[1]} of '

    def _get_entity_tokenize_idx(self, input_ids, entity):
        idx = self.tokenizer(" "+entity, add_special_tokens=False)['input_ids'][0]
        input_ids = input_ids.view(-1)
        iddd = (input_ids == idx).nonzero(as_tuple=True)[0]

        return iddd






    def _load_symbol2id(self):
        # gen symbol2id, without embedding
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        # rel and ent combine together
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def __len__(self):
        return len(self.samples)

class CustomSampler(Sampler):
    """
    For the batch size custom
    return 1 per batch
    """
    def __init__(self, data):
        super().__init__(self)
        self.data = data
    
    def __iter__(self):
        indices = torch.arange(0, len(self.data))
        indices = indices[torch.randperm(indices.shape[0])].unsqueeze(1)

        return iter(indices)


    def __len__(self):
        return len(self.data)


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

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
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
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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
        return ["0", "1"]

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
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]#.find(',')
                    ent2text[temp[0]] = temp[1]#[:end]
  
        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    #first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]#[:first_sent_end_position + 1] 

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]      

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []




        for (i, line) in enumerate(lines):
            
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":

                label = "1"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
                
            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(5):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break                    
                        tmp_head_text = ent2text[tmp_head]
                        examples.append(
                            InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))       
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(5):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[tmp_tail]
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))                                                  
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, args):
    """Loads a data file into a list of `InputBatch`s."""
    print_info = True
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        


        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_c = tokenizer.tokenize(example.text_c)

        _truncate_seq_pair(tokens_a, tokens_c, 56)

        example.text_a = tokenizer.convert_tokens_to_string(tokens_a)
        example.text_c = tokenizer.convert_tokens_to_string(tokens_c)


        """
        for WN18RR : <head> <rel> ? <sep> <mask> , I think that <tail> .
        for umsl : Is that <head> <rel> <tail> ? <mask> .
        """

        # prompt_inputs = " ".join([example.text_a, example.text_b, f"? , I think that {example.text_c} ."])
        # import IPython; IPython.embed(); exit(1)

        if args.prompt:
            if "WN18RR" in args.data_dir:
                inputs = tokenizer(
                    f"{example.text_a} {example.text_b} ?",
                    f"{tokenizer.mask_token}, I think that {example.text_c} .",
                    truncation="longest_first",
                    max_length=max_seq_length,
                    padding="max_length",
                    add_special_tokens=True
                )
            else:
                inputs = tokenizer(
                    f"Is that {example.text_a} {example.text_b} {example.text_c} ? ",
                    f"{tokenizer.mask_token} .",
                    truncation="longest_first",
                    max_length=max_seq_length,
                    padding="max_length",
                    add_special_tokens=True
                )
            assert 103 in inputs['input_ids'], "[MASK] in the sentence"
        else:
            inputs = tokenizer(
                f"{example.text_a} {example.text_b}",
                f"{example.text_c}",
                truncation="longest_first",
                max_length=max_seq_length,
                padding="max_length",
                add_special_tokens=True
            )



        label_id = label_map[example.label]


        features.append(
                InputFeatures(input_ids=inputs["input_ids"],
                              input_mask=inputs['attention_mask'],
                              segment_ids=inputs['token_type_ids'],
                              label_id=label_id))
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


def get_dataset(args, processor, label_list, tokenizer, mode):

    assert mode in ["train", "dev", "test"], "mode must be in train dev test!"

    if mode == "train":
        train_examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        train_examples = processor.get_dev_examples(args.data_dir)
    else:
        train_examples = processor.get_test_examples(args.data_dir, args.chunk)




    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, args=args)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return train_data

if __name__ == "__main__":
    dataset = KGCDataset('./dataset')