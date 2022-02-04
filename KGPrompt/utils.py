import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from data import convert_examples_to_features
from data import KGProcessor

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
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines




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

logger = logging.getLogger()



# TODO write a dataset for fast test processing

class TestDataset(Dataset):
    def __init__(self, args, test_triples, tokenizer, processor):
        self.test_triples = test_triples
        self.tokenizer = tokenizer
        self.processor = processor
        self.args = args

        self.label_list = processor.get_labels(args.data_dir)


        self.entity_list = processor.get_entities(args.data_dir)

    def __len__(self):
        return len(self.test_triples)
    
    def __getitem__(self, index):
        entity_list = self.entity_list
        all_triples_str_set = set()
        processor = self.processor
        args = self.args 
        tokenizer = self.tokenizer
        label_list = self.label_list

        test_triple = self.test_triples[index]
        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        # print(test_triple, head, relation, tail)

        head_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = "\t".join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)
        
        tmp_examples = processor._create_examples(
            head_corrupt_list, "test", args.data_dir
        )
        # print(len(tmp_examples))
        tmp_features = convert_examples_to_features(
            tmp_examples, label_list, args.max_seq_length, tokenizer, args
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in tmp_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in tmp_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in tmp_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in tmp_features], dtype=torch.long
        )

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        # Run prediction for temp data
        eval_sampler = SequentialSampler(eval_data)
        left_eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=16
        )

        tail_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = "\t".join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)

        tmp_examples = processor._create_examples(
            tail_corrupt_list, "test", args.data_dir
        )
        # print(len(tmp_examples))
        tmp_features = convert_examples_to_features(
            tmp_examples, label_list, args.max_seq_length, tokenizer, args
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in tmp_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in tmp_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in tmp_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in tmp_features], dtype=torch.long
        )

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        # Run prediction for temp data
        eval_sampler = SequentialSampler(eval_data)
        right_eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=16
        )

        return dict(left=left_eval_dataloader, right=right_eval_dataloader)




def test_model(args, model, tokenizer, wandb_logger):
    model.eval()
    processor = KGProcessor()
    # get the chunk entities
    test_triples = processor.get_test_triples(args.data_dir, args.chunk)
    dataset = TestDataset(args, test_triples,tokenizer=tokenizer, processor=processor)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=4, collate_fn=lambda x:x)
    all_triples_str_set = set()
    # get all the entities
    entity_list = processor.get_entities(args.data_dir)
    label_list = processor.get_labels(args.data_dir)
    device = torch.device("cuda:0")

    model = model.to(device)

    ranks = []
    ranks_left = []
    ranks_right = []

    hits_left = []
    hits_right = []
    hits = []

    top_ten_hit_count = 0

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    pbar = tqdm(total=len(test_triples), desc="Testing...")

    for batch in dataloader:
        left_dataloader = batch[0]['left']
        right_dataloader = batch[0]['right']
        preds = []
        for input_ids, input_mask, segment_ids, label_ids in left_dataloader:

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(
                    input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                )
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)

            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        # print(preds, preds.shape)
        rel_values = preds[:, 1]
        rel_values = torch.tensor(rel_values)
        # print(rel_values, rel_values.shape)
        _, argsort1 = torch.sort(rel_values, descending=True)
        # print(max_values)
        # print(argsort1)
        argsort1 = argsort1.cpu().numpy()
        rank1 = np.where(argsort1 == 0)[0][0]
        # print("left: ", rank1)
        ranks.append(rank1 + 1)
        ranks_left.append(rank1 + 1)
        if rank1 < 10:
            top_ten_hit_count += 1

        preds = []
        for input_ids, input_mask, segment_ids, label_ids in right_dataloader:

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(
                    input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                )
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)

            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        rel_values = preds[:, 1]
        rel_values = torch.tensor(rel_values)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        rank2 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank2 + 1)
        ranks_right.append(rank2 + 1)
        # print("right: ", rank2)
        # print("mean rank until now: ", np.mean(ranks))
        if rank2 < 10:
            top_ten_hit_count += 1
        
        for hits_level in range(10):
            if rank1 <= hits_level:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if rank2 <= hits_level:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
        
        pbar.update(1)
        pbar.set_postfix({"mean rank": np.mean(ranks), "hit@10": top_ten_hit_count * 1.0 / len(ranks) })

    
    if args.chunk:
        with open(f"chuck{args.chunk}_result_rank.pkl", "wb") as file:
            pickle.dump(ranks, file)


    print(f"mean rank: {np.mean(ranks)} \nhits@10: {top_ten_hit_count * 1.0 / len(ranks)}")



def _test_model(args, model, tokenizer, wandb_logger):
    # run link prediction
    # only use one gpu



    processor = KGProcessor()
    # get the chunk entities
    test_triples = processor.get_test_triples(args.data_dir, args.chunk)
    dataset = TestDataset(args, test_triples,tokenizer=tokenizer, processor=processor)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=2)
    all_triples_str_set = set()
    # get all the entities
    entity_list = processor.get_entities(args.data_dir)
    label_list = processor.get_labels(args.data_dir)
    device = torch.device("cuda:0")

    model = model.to(device)

    ranks = []
    ranks_left = []
    ranks_right = []

    hits_left = []
    hits_right = []
    hits = []

    top_ten_hit_count = 0

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    pbar = tqdm(total=len(test_triples), desc="Testing...")

    for test_triple in test_triples:
        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        # print(test_triple, head, relation, tail)

        head_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = "\t".join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)

        tmp_examples = processor._create_examples(
            head_corrupt_list, "test", args.data_dir
        )
        # print(len(tmp_examples))
        tmp_features = convert_examples_to_features(
            tmp_examples, label_list, args.max_seq_length, tokenizer, args
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in tmp_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in tmp_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in tmp_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in tmp_features], dtype=torch.long
        )

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        # Run prediction for temp data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=16
        )
        model.eval()

        preds = []

        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(
                    input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                )
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)

            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        # print(preds, preds.shape)
        rel_values = preds[:, all_label_ids[0]]
        rel_values = torch.tensor(rel_values)
        # print(rel_values, rel_values.shape)
        _, argsort1 = torch.sort(rel_values, descending=True)
        # print(max_values)
        # print(argsort1)
        argsort1 = argsort1.cpu().numpy()
        rank1 = np.where(argsort1 == 0)[0][0]
        # print("left: ", rank1)
        ranks.append(rank1 + 1)
        ranks_left.append(rank1 + 1)
        if rank1 < 10:
            top_ten_hit_count += 1

        tail_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = "\t".join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)

        tmp_examples = processor._create_examples(
            tail_corrupt_list, "test", args.data_dir
        )
        # print(len(tmp_examples))
        tmp_features = convert_examples_to_features(
            tmp_examples, label_list, args.max_seq_length, tokenizer, args
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in tmp_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in tmp_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in tmp_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in tmp_features], dtype=torch.long
        )

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        # Run prediction for temp data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=16
        )
        model.eval()
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(
                    input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                )
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)

            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        rel_values = preds[:, all_label_ids[0]]
        rel_values = torch.tensor(rel_values)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        rank2 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank2 + 1)
        ranks_right.append(rank2 + 1)
        # print("right: ", rank2)
        # print("mean rank until now: ", np.mean(ranks))
        if rank2 < 10:
            top_ten_hit_count += 1
        # print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

        pbar.update(1)
        pbar.set_postfix({"mean rank": np.mean(ranks), "hit@10": top_ten_hit_count * 1.0 / len(ranks) })

        # file_prefix = (
        #     str(args.data_dir[7:])
        #     + "_"
        #     + str(args.batch_size)
        #     + "_"
        #     + str(args.lr)
        #     + "_"
        #     + str(args.max_seq_length)
        #     + "_"
        #     + str(args.max_epochs)
        # )
        # # file_prefix = str(args.data_dir[7:])
        # f = open(file_prefix + "_ranks.txt", "a")
        # f.write(str(rank1) + "\t" + str(rank2) + "\n")
        # f.close()
        # this could be done more elegantly, but here you go
        for hits_level in range(10):
            if rank1 <= hits_level:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if rank2 <= hits_level:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
        

    for i in [0, 2, 9]:
        logger.info("Hits left @{0}: {1}".format(i + 1, np.mean(hits_left[i])))
        logger.info("Hits right @{0}: {1}".format(i + 1, np.mean(hits_right[i])))
        logger.info("Hits @{0}: {1}".format(i + 1, np.mean(hits[i])))
        wandb_logger.log_metrics({f'hits{i+1}': np.mean(hits[i])})
    logger.info("Mean rank left: {0}".format(np.mean(ranks_left)))
    logger.info("Mean rank right: {0}".format(np.mean(ranks_right)))
    logger.info("Mean rank: {0}".format(np.mean(ranks)))
    logger.info(
        "Mean reciprocal rank left: {0}".format(np.mean(1.0 / np.array(ranks_left)))
    )
    logger.info(
        "Mean reciprocal rank right: {0}".format(np.mean(1.0 / np.array(ranks_right)))
    )
    logger.info("Mean reciprocal rank: {0}".format(np.mean(1.0 / np.array(ranks))))
    wandb_logger.log_metrics({'mrr': np.mean(1.0 / np.array(ranks))})
    wandb_logger.log_metrics({'mr': np.mean(ranks)})


    if args.chunk:
        with open(f"chuck{args.chunk}_result_rank.pkl", "wb") as file:
            pickle.dump(ranks, file)


def gather_all_ranks():
    ranks = np.array([])
    for i in range(10):
        with open(f"chuck{i}_result_rank.pkl", "rb") as file:
            ranks = np.concatenate([ranks, pickle.load(file)], axis=0)
    
    return ranks.mean(), (ranks<=10).mean()
    
