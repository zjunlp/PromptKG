import json
import logging
import pandas as pd
import numpy as np
import dataclasses
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers.data import InputFeatures
from typing import List, Optional, Union

from src.data_processor import processors_mapping, median_mapping
from src.utils import count_special_tokens_in_template
from src.tokenizer import tokenize_multipart_input

logger = logging.getLogger(__name__)


class FewShotDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train', demo=False):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer 
        self.mode = mode

        self.demo = demo
        if self.demo:
            logger.info("Use demontrations")
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        # only for prompt tuning
        self.total_label_tokens = 0

        self.len_special_tokens_in_template = 0
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)
            self.label2id = {key: idx for idx, key in enumerate(self.label_to_word)}
            self.max_num_tokens_in_label = 1

            for key in self.label_to_word:
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))
        
            if self.num_labels> 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]

            self.len_special_tokens_in_template = count_special_tokens_in_template(
                self.args.template, tokenizer=tokenizer, max_len_label_tokens=self.max_num_tokens_in_label,
            )
        else:
            self.label_to_word = None
            self.label_word_list = None

        logger.info(f"Length of special tokens in template: {self.len_special_tokens_in_template}")
        logger.info(f"Creating examples from dataset file at {args.data_dir}")
        self.support_examples = self.processor.get_train_examples(args.data_dir)

        if mode == 'dev':
            self.query_examples = self.processor.get_dev_examples(args.data_dir)
        elif mode == 'test':
            self.query_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.query_examples = self.support_examples

        self.size = len(self.query_examples)
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for query_idx in range(len(self.query_examples)):
            context_indices = [support_idx for support_idx in support_indices if support_idx != query_idx or mode != "train"]
            self.example_idx.append((query_idx, context_indices))
        
        if mode != 'train':
            self.features = []
            for query_idx, context_indices in self.example_idx:
                example = self.query_examples[query_idx]
                template = args.template
                demo_template = args.demo_template
                supports = self.select_context([self.support_examples[i] for i in context_indices])

                self.features.append(self.convert_fn(
                    example=example,
                    supports=supports,
                    label_list=self.label_list,
                    prompt=args.prompt,
                    template=template,
                    demo_template=demo_template,
                    label_word_list=self.label_word_list,
                ))

        else:
            self.features = None

    def select_context(self, context_examples):
        """
        Select demonstration from provided examples.
        """
        max_demo_per_label = 1
        counts = {k: 0 for k in self.label_list}
        if len(self.label_list) == 1:
            # Regression
            counts = {'0': 0, '1': 0}

        selection = []
        order = np.random.permutation(len(context_examples))
        for i in order:
            label = context_examples[i].label
            if len(self.label_list) == 1:
                # Regression
                label = '0' if float(label) <= median_mapping[self.args.task_name] else '1'
            if counts[label] < max_demo_per_label:
                selection.append(context_examples[i])
                counts[label] += 1
            if sum(counts.values()) == len(counts) * max_demo_per_label:
                break

        return selection

    def __len__(self):
        return self.size

    def get_labels(self):
        return self.label_list

    def __getitem__(self, index):
        if self.features is None:
            query_idx, context_indices = self.example_idx[index]
            # The input (query) example
            example = self.query_examples[query_idx]
            supports = self.select_context([self.support_examples[i] for i in context_indices])
            template = self.args.template
            demo_template = self.args.demo_template

            features = self.convert_fn(
                example=example,
                supports=supports,
                label_list=self.label_list,
                prompt=self.args.prompt,
                template=template,
                demo_template=demo_template,
                label_word_list=self.label_word_list,
            )

            return features
        else:
            features = self.features[index]
            return features

    def convert_fn(
        self,
        example,
        supports,
        label_list=None,
        prompt=False,
        template=None,
        demo_template=None,
        label_word_list=None,
    ):
        """
        return a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length
        demo_max_length = self.args.demo_max_length
        demo_first_sent_limit = self.args.demo_first_sent_limit
        demo_other_sent_limit = self.args.demo_other_sent_limit

        label_map = {label: i for i, label in enumerate(label_list)}
        
        num_seq_per_example = 1
        if example.text_b is not None:
            num_seq_per_example = 2

        # convert label to integer
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]
        
        if not self.demo:
            inputs, _ = tokenize_multipart_input(
                input_text_list=input_example_to_tuple(example),
                max_length=max_length,
                tokenizer=self.tokenizer,
                prompt=prompt,
                len_special_tokens_in_template=self.len_special_tokens_in_template,
                virtual_demo=self.args.virtual_demo,
                virtual_demo_length_per_label=self.args.virtual_demo_length_per_label,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                demo=False,
                num_seq_per_example=num_seq_per_example
            )
            features = OurInputFeatures(**inputs, label=example_label)
        else:            
            augmented_example = []
            query_text = input_example_to_tuple(example)
            support_by_label = [[] for i in range(len(label_map))]
        
            for label_name, label_id in label_map.items():
                if len(label_list) == 1:
                    # Regression
                    for support_example in filter(lambda s: ('0' if float(s.label) <= median_mapping[self.args.task_name] else '1') == label_name, supports):
                        support_by_label[label_id] += input_example_to_tuple(support_example)
                else:
                    for support_example in filter(lambda s: s.label == label_name, supports):
                        support_by_label[label_id] += input_example_to_tuple(support_example)

            augmented_example = query_text
            for label_id in range(len(label_map)):
                augmented_example += support_by_label[label_id]

            inputs, demo_result = tokenize_multipart_input(
                input_text_list=augmented_example,
                max_length=max_length,
                tokenizer=self.tokenizer,
                prompt=prompt,
                len_special_tokens_in_template=self.len_special_tokens_in_template,
                virtual_demo=self.args.virtual_demo,
                virtual_demo_length_per_label=self.args.virtual_demo_length_per_label,
                template=template,
                demo_template=demo_template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                demo=True,
                demo_max_length=demo_max_length,
                demo_first_sent_limit=demo_first_sent_limit,
                demo_other_sent_limit=demo_other_sent_limit,
                num_seq_per_example=num_seq_per_example
            )
            features = OurInputFeatures(**inputs, label=example_label)
            if self.mode == 'train' and self.args.virtual_demo:
                demo_features = [OurInputFeatures(**inputs_, label=example_label) for inputs_ in demo_result]
                return (features, demo_features)
        return features

def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)
    block_flag_for_demo: Optional[List[int]] = None
    block_flag_for_prompt: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"
