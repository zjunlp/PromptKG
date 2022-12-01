from typing import List
from collections import Counter
from multiprocessing import Pool
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
"""
这个文件负责处理从
1. load dataset file into object
2. process the object into batch tensor

QA 数据集
1. train dev test
2. entity2text
3. relation2text
"""
@dataclass
class KBQAExample:
    """
    Question:  what does [Jay Hernandez] act in	
    Triples:

    <[Jay Hernandez] [act] [Hostel]>

    """
    question: str
    triples: list 


class KBQADataset(Dataset):
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
        with open(f"./dataset/{self.args.dataset}/{self.args.k_hop}-hop/{self.mode}.json") as file:
            data = [json.loads(_) for _ in file.readlines()]
        return data
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]