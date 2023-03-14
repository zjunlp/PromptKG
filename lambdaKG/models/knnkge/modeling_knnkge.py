from transformers import BertForMaskedLM
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

class KNNKGEModel(BertForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser




