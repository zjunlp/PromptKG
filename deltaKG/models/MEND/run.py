
import copy
import random
import importlib
import logging
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils
from argparse import ArgumentParser

from trainer import EditTrainer
import models
import warnings, csv, sys
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)

# @hydra.main(config_path='config', config_name='config')
def run():
    # print(config)
    # print(type(config))
    with open(config_file, 'r') as f:
        # OmegaConf.save(config=config, f=f.name)
        config = OmegaConf.load(f)

    config.data.n_edits = num_edits

    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    # base_dir = hydra.utils.get_original_cwd()
    # LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)

    if config.task == "gen" or config.task == "wiki":
        add_padding(tokenizer, model)
        from data_classes.wiki import GenDataset

        train_set = GenDataset("train", tokenizer, config, config.data.path, pct=10)
        val_set = GenDataset("validation", tokenizer, config, config.data.path, pct=10)
    elif config.task == "kgc":
        from data_classes.kgc import KGC
        # config may not match
        data = KGC(config)
        data.setup()
        config.__dict__.update(data.get_config())
        # train/val set has a func named edit_generator
        train_set = data.data_train
        val_set = data.data_val
        test_set = data.data_test
    else:
        raise ValueError(f"Unrecognized task {config.task}")

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, data, lambda: copy.deepcopy(model)) # model存到了mend

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(config.ft.locality.batch_size + 1)
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(train_set.edit_generator(config.ft.locality.batch_size + 1))["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    trainer = EditTrainer(alg, config, train_set, val_set, data.sampler.memory)
    trainer.run()
    trainer.validate()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_edits", type=int, default=8)
    parser.add_argument("--config_file", type=str, default=None)

    args = parser.parse_args()
    global num_edits
    global config_file
    num_edits = args.n_edits
    config_file = args.config_file
    run()
