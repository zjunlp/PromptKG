import math
import numpy as np
import torch
from collections import defaultdict
from typing import Callable, Dict, Optional, List, Tuple
from transformers.trainer_utils import EvalPrediction

from src.data_processor import compute_metrics_mapping


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        logits = p.predictions
        preds = np.argmax(logits, axis=1)
        label_ids = p.label_ids

        return compute_metrics_mapping[task_name](task_name, preds, label_ids)
    
    return compute_metrics_fn


def data_collator_for_cl(features):
    num_instance = len(features[0]) if isinstance(features[0], Tuple) else -1 

    if num_instance == 2:
        features1 = [vars(f[0]) for f in features]
        features2 = []
        
        for f in features:
            features2.extend([vars(f_) for f_ in f[1]])
    else:
        features1 = [vars(f) for f in features]

    first = features1[0]
    batch1 = {}
    batch2 = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch1["labels"] = torch.tensor([f["label"] for f in features1], dtype=dtype)
        if num_instance == 2:
            batch2["labels"] = torch.tensor([f["label"] for f in features2], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch1["labels"] = torch.stack([f["label_ids"] for f in features1])
            if num_instance == 2:
                batch2["labels"] = torch.stack([f["label_ids"] for f in features2])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch1["labels"] = torch.tensor([f["label_ids"] for f in features1], dtype=dtype)
            if num_instance == 2:
                batch2["labels"] = torch.tensor([f["label_ids"] for f in features2], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch1[k] = torch.stack([f[k] for f in features1])
                if num_instance == 2:
                    batch2[k] = torch.stack([f[k] for f in features2])
            else:
                batch1[k] = torch.tensor([f[k] for f in features1])
                if num_instance == 2:
                    batch2[k] = torch.tensor([f[k] for f in features2])

    if num_instance == 2:
        return {'v1': batch1, 'v2': batch2}
    
    return batch1

def count_special_tokens_in_template(
    template,
    tokenizer,
    max_len_label_tokens
):
    len_special_token_in_template = 0

    special_token_mapping = {
        'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id, 
        'prompt': tokenizer.pad_token_id
    }

    for part in template.split("*"):
        if part in special_token_mapping:
            if part == 'mask':
                len_special_token_in_template += max_len_label_tokens
            else:
                len_special_token_in_template += 1
        elif 'sent' in part:
            continue
        else:
            # Just natural language prompt
            part = part.replace('_', ' ') 
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                len_special_token_in_template += 1
            else:
                len_special_token_in_template += len(tokenizer.encode(part, add_special_tokens=False))

    return len_special_token_in_template

def map_supports(supports):
    examples_by_class = defaultdict(lambda: list())
    for example in supports:
        examples_by_class[example.label].append(example)

    feature_by_class = {}
    for label in examples_by_class.keys():
        batch = {}

        examples = examples_by_class[label]
        features = [vars(e) for e in examples]
        first = features[0]
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)

        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        feature_by_class[label] = batch
    
    return feature_by_class
            