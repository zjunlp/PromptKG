from dataclasses import dataclass, asdict
from typing import Optional, Union
from enum import Enum
import random
import numpy as np
import json

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, BertTokenizer
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)

from .processor import process_mapping, get_dataset

import pytorch_lightning as pl
transformers.logging.set_verbosity_error()



class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args=None) -> None:
        super().__init__()
        self.args = args
        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self, stage=None):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.data_train = None
        self.data_val = None
        self.data_test = None
        
    def get_config(self):
        d = {}
        for k, v in self.__dict__.items():
            if "st" in k or "ed" in k:
              d.update({k: v})
            elif k == "data_train" or k == "data_val" or k == "data_test":
              d.update({k: v})
            elif k == "sampler" or k == "tokenizer":
              d.update({k: v})

        return d

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)


@dataclass
class DataCollatorForKE:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    num_labels: int = 0
    stable_batch_size: int = None
    memory: list = None
    memory_perm: list = None
    memory_pos: int = 0
    task_name: str = None

    def __call__(self, features, return_tensors=None):
        cur_features = {}
        
        if self.memory != None:
            ori_batch = [self.memory.__getitem__(i) for i in self.memory_perm[self.memory_pos:self.memory_pos+self.stable_batch_size]]
            self.memory_pos += self.stable_batch_size
            features = ori_batch + features
        
        label = [feature["label"] for feature in features]
        labels = [feature["labels"][0] for feature in features]
        
        cur_features['labels'] = torch.tensor(labels, dtype=torch.int64)
        cur_features['label'] = torch.tensor(label, dtype=torch.int64)
        if return_tensors is None:
            return_tensors = self.return_tensors

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        cur_features['input_ids'] = features['input_ids']
        cur_features['attention_mask'] = features['attention_mask']
        cur_features['cond_input_ids'] = features['cond_input_ids']
        cur_features['cond_attention_mask'] = features['cond_attention_mask']

        return cur_features
      
@dataclass
class DataCollatorForCaliNet:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    num_labels: int = 0
    stable_batch_size: int = None
    memory: list = None
    memory_perm: list = None
    memory_pos: int = 0
    task_name: str = None

    def __call__(self, features, return_tensors=None):
        cur_features = {}
        
        if self.memory != None:
            ori_batch = [self.memory.__getitem__(i) for i in self.memory_perm[self.memory_pos:self.memory_pos+self.stable_batch_size]]
            self.memory_pos += self.stable_batch_size
            features += ori_batch
            
        label = [feature["label"] for feature in features]
        labels = [feature["labels"][0] for feature in features]
        
        cur_features['labels'] = torch.tensor(label, dtype=torch.int64)
        cur_features['label'] = torch.tensor(labels, dtype=torch.int64)
        if return_tensors is None:
            return_tensors = self.return_tensors

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        cur_features['input_ids'] = features['input_ids']
        cur_features['attention_mask'] = features['attention_mask']

        return cur_features


class KEKGC(BaseDataModule):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       use_fast=False)
        self.processor = process_mapping[args.kge_model_type](self.tokenizer, args)
        self.label_list = self.processor.get_labels(args.data_dir)

        entity_list = self.processor.get_entities(args.data_dir)

        num_added_tokens = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': entity_list})
        self.sampler = DataCollatorForKE(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8,
            padding="longest",
            max_length=args.max_seq_length,
            num_labels=len(entity_list),
            task_name=args.task_name,
            stable_batch_size=args.stable_batch_size
        )
        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)
        num_added_tokens = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': relations_tokens})

        vocab = self.tokenizer.get_added_vocab()
        self.relation_id_st = vocab[relations_tokens[0]]
        self.relation_id_ed = vocab[relations_tokens[-1]] + 1
        self.entity_id_st = vocab[entity_list[0]]
        self.entity_id_ed = vocab[entity_list[-1]] + 1
        self.rng = np.random.default_rng(args.seed)

    def setup(self, stage=None):
        self.data_train = get_dataset(self.args, self.processor,
                                      self.label_list, self.tokenizer, "train")
        
        self.data_val = get_dataset(self.args, self.processor, self.label_list,
                                    self.tokenizer, "dev")

        self.sampler.memory = get_dataset(self.args, self.processor,
                                      self.label_list, self.tokenizer, "memory") if self.args.task_name == 'add' or self.args.task_name == 'edit' else None

        self.sampler.memory_perm = self.rng.permutation(len(self.sampler.memory)).tolist() if self.sampler.memory != None else None
        
class CaliNetKGC(BaseDataModule):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       use_fast=False)
        self.processor = process_mapping[args.kge_model_type](self.tokenizer, args)
        self.label_list = self.processor.get_labels(args.data_dir)

        entity_list = self.processor.get_entities(args.data_dir)

        num_added_tokens = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': entity_list})
        self.sampler = DataCollatorForCaliNet(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8,
            padding="longest",
            max_length=args.max_seq_length,
            num_labels=len(entity_list),
            task_name=args.task_name,
            stable_batch_size=args.stable_batch_size
        )
        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)
        num_added_tokens = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': relations_tokens})

        vocab = self.tokenizer.get_added_vocab()
        self.relation_id_st = vocab[relations_tokens[0]]
        self.relation_id_ed = vocab[relations_tokens[-1]] + 1
        self.entity_id_st = vocab[entity_list[0]]
        self.entity_id_ed = vocab[entity_list[-1]] + 1
        self.rng = np.random.default_rng(args.seed)

    def setup(self, stage=None):
        self.data_train = get_dataset(self.args, self.processor,
                                      self.label_list, self.tokenizer, "train")
        
        self.data_val = get_dataset(self.args, self.processor, self.label_list,
                                    self.tokenizer, "dev")

        self.sampler.memory = get_dataset(self.args, self.processor,
                                      self.label_list, self.tokenizer, "memory") if self.args.task_name == 'add' or self.args.task_name == 'edit' else None

        self.sampler.memory_perm = self.rng.permutation(len(self.sampler.memory)).tolist() if self.sampler.memory != None else None



dataset_mapping = {
  'KE': KEKGC,
  'KGEditor': KEKGC,
  'CaliNet': CaliNetKGC,
  'K-Adapter': CaliNetKGC,
}