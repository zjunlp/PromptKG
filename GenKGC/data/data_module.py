from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from enum import Enum
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
# from transformers.configuration_bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)

from .base_data_module import BaseDataModule
from .processor import KGProcessor, get_dataset


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


@dataclass
class DataCollatorForSeq2Seq:
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
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        
        ent = [_.pop("filter_ent_ids") for _ in features]
        input_sentences = None
        input_sentences = [_.pop("input_sentences") for _ in features]
        relation_ids = [_.pop("relation_ids") for _ in features]
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        features["filter_ent_ids"] = ent
        if input_sentences[0]:
            features["input_sentences"] = input_sentences

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features



class KGC(BaseDataModule):
    def __init__(self, args, model) -> None:
        super().__init__(args)
        if "mbart" in args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, add_prefix_space=True, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
        else:
            if "Ali" in args.data_dir:
                self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path, add_prefix_space=True)
                self.tokenizer.bos_token = self.tokenizer.cls_token
                self.tokenizer.eos_token = self.tokenizer.sep_token
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, add_prefix_space=True)
            
        self.processor = KGProcessor()
        self.label_list = self.processor.get_labels(args.data_dir)
        
        spo_list = ["(reverse)"]
        if spo_list[0] not in self.tokenizer.additional_special_tokens:
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': spo_list})
        
        # fix
        relations_ids = ["[RELATION_{i}]" for i in range(len(self.label_list))]
        if spo_list[0] not in self.tokenizer.additional_special_tokens:
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relations_ids})
        self.sampler = DataCollatorForSeq2Seq(self.tokenizer,
            model=model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.args.precision == 16 else None,
            padding="longest",
            max_length=self.args.max_seq_length
        )

    def setup(self, stage=None):
        self.data_train = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "train")
        self.data_val = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "dev")
        self.data_test = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "test")

    def prepare_data(self):
        pass


    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--data_dir", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--use_demos", type=int, default=0)
        return parser

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=self.num_workers, pin_memory=True, collate_fn=self.sampler, batch_size=self.args.batch_size, shuffle=not ("YAGO" in self.args.data_dir or "Ali" in self.args.data_dir))

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=self.num_workers, pin_memory=True, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, num_workers=self.num_workers, pin_memory=True, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)



class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.pre_whole_word = None
        self.mask_whole_word = None
        self.vocab_words = None
        self.call_count = 0
        self.offline_mode = False
        self.skipgram_size_geo_list = None
        self.span_same_mask = False

    def init_skipgram_size_geo_list(self, p):
        if p > 0:
            g_list = []
            t = p
            for _ in range(self.skipgram_size):
                g_list.append(t)
                t *= (1-p)
            s = sum(g_list)
            self.skipgram_size_geo_list = [x/s for x in g_list]

    def __call__(self, instance):
        raise NotImplementedError
