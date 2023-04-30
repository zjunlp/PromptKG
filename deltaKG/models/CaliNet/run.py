#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,m software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import math
import logging
import os
import sys
sys.path.append("..")
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import warnings
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
sys.path.append(os.getcwd())
from src.data.data_module import dataset_mapping

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    EvalPrediction
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

sys.path.append(os.getcwd())
from src.models.modeling_bert import bert_mapping
from src.models.trainer import trainer_mapping
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
os.environ["WANDB_MODE"] = "offline"
check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to tokenizer from huggingface.co/models"}
    )
    model_checkpoint: str = field(
        metadata={"help": "Path to pretrained model"}
    )
    kge_model_type: str = field(
        default='CaliNet', metadata={"help": "Model type which decides the type of trainer"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    kb_layer: str = field(
        default="",
        metadata={"help": "Layers to be extended, split by a comma, e.g., '22,23'"},
    )
    ex_size: int = field(
        default="",
        metadata={"help": "Layers to be extended, split by a comma, e.g., '22,23'"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    # train_file: Optional[str] = field(
    #     default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    # )
    # validation_file: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
    #         "(a jsonlines or csv file)."
    #     },
    # )
    # test_file: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
    #     },
    # )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    max_seq_length: int = field(
        default=64,
        metadata={
            "help": "max seq length"
        },
    )
    stable_batch_size: int = field(
        default=64,
        metadata={
            "help": "stable_batch_size"
        },
    )
    data_dir: str = field(
        default="../data/EditKnowledge",
        metadata={
            "help": "data_dit"
        },
    )
    batch_size: int = field(
        default=8,
        metadata={
            "help": "batch_size"
        },
    )
    eval_batch_size: int = field(
        default=8,
        metadata={
            "help": "eval batch_size"
        },
    )
    pretrain: int = field(
        default=0,
        metadata={
            "help": "pretrain"
        },
    )
    num_workers: int = field(
        default=8,
        metadata={
            "help": "pretrain"
        },
    )
    task_name: str = field(
        default="edit",
        metadata={
            "help": "task name"
        },
    )
    save_model_name: str = field(
        default="fb15k_237_edit",
        metadata={
            "help": "task name"
        },
    )

    def __post_init__(self):
        return
        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #     return
            # raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        # if self.val_max_target_length is None:
        #     self.val_max_target_length = self.max_target_length

def compute_edit(all_edit_ranks, all_stable_ranks):
    edit_ranks = np.array(all_edit_ranks)
    loc_ranks = np.array(all_stable_ranks)

    loc_hits5 = (loc_ranks<=5).mean()
    loc_hits3 = (loc_ranks<=3).mean()
    loc_hits1 = (loc_ranks<=1).mean()
    edit_hits5 = (edit_ranks<=5).mean()
    edit_hits3 = (edit_ranks<=3).mean()
    edit_hits1 = (edit_ranks<=1).mean()
    logger.info(f"Eval/hits5: {edit_hits5}")
    logger.info(f"Eval/hits3: {edit_hits3}")
    logger.info(f"Eval/hits1: {edit_hits1}")
    logger.info(f"Eval/mean_rank: {edit_ranks.mean()}")
    logger.info(f"Eval/mrr: {(1. / edit_ranks).mean()}")
    logger.info(f"Eval/loc_hits1: {loc_hits1}")
    logger.info(f"Eval/loc_hits3: {loc_hits3}")
    logger.info(f"Eval/loc_hits5: {loc_hits5}")
    logger.info(f"Eval/loc_mean_rank: {loc_ranks.mean()}")
    logger.info(f"Eval/loc_mrr: {(1. / loc_ranks).mean()}")

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # TODO cleaner method
    # training_args['label_names'] = ["label"]
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_args.model_name_or_path = model_args.model_name_or_path
    data_args.kge_model_type = model_args.kge_model_type
    data_args.seed = training_args.seed
    # training_args.stable_batch_size = data_args.stable_batch_size
    # add KGC dataset
    kgc_data = dataset_mapping[model_args.kge_model_type](data_args)
    kgc_data.setup()
    global config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_checkpoint,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.ex_size = model_args.ex_size
    config.kb_layer = model_args.kb_layer
    config.decoder_start_token_id = kgc_data.entity_id_st
    config.decoder_end_token_id = kgc_data.entity_id_ed
    
    tokenizer = kgc_data.tokenizer
    config.mask_token_id = kgc_data.tokenizer.mask_token_id
    model = bert_mapping[model_args.kge_model_type].from_pretrained(
        model_args.model_checkpoint,
        from_tf=bool(".ckpt" in model_args.model_checkpoint),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if training_args.do_train:
        train_dataset = kgc_data.data_train

    if training_args.do_eval:
        eval_dataset = kgc_data.data_val

    # Initialize our Trainer
    training_args.num_workers = data_args.num_workers
    kgc_trainer = trainer_mapping[model_args.kge_model_type](
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        stable_dataset=kgc_data.sampler.memory,
        tokenizer=tokenizer,
        data_collator=kgc_data.sampler,
        compute_metrics=compute_edit,
        stable_batch_size=data_args.stable_batch_size
    )
    kgc_trainer.__dict__.update(kgc_data.get_config())
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = kgc_trainer.train(resume_from_checkpoint=checkpoint)

if __name__ == "__main__":
    main()
