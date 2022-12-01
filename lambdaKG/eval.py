import argparse
import importlib
from logging import debug

import numpy as np
import torch
import fcntl
import pytorch_lightning as pl
import lit_models
import yaml
import time
from transformers import AutoConfig
import os
from models import Trie

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--lit_model_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="KGC")
    parser.add_argument("--model_class", type=str, default="RobertaUseLabelWord")
    parser.add_argument("--checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    lit_model_class = _import_class(f"lit_models.{temp_args.lit_model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    if hasattr(model_class, "add_to_argparse"):
        model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

metric_list = {"knnkge" : "hits10",
                "simkgc" : "acc1",
                "t5kbqa" : "hits1"}

def main():
    parser = _setup_parser()
    args = parser.parse_args()

    # use all the available gpus
    args.gpus = torch.cuda.device_count()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    data_class = _import_class(f"data.{args.data_class}")
    litmodel_class = _import_class(f"lit_models.{args.lit_model_class}")

    # config = AutoConfig.from_pretrained(args.model_name_or_path)
    # # update parameters
    # config.label_smoothing = args.label_smoothing
    

    # model = model_class.from_pretrained(args.model_name_or_path, config=config)
    # perfered , warp the transformers encoder
    method_name = args.model_class.lower().replace("model","")
    if method_name in metric_list:
        metric_name = metric_list[method_name]
    else:
        metric_name = "hits10"
    # model = model_class(args) if method
    data = data_class(args)
    tokenizer = data.tokenizer

    lit_model = litmodel_class(args=args, tokenizer=tokenizer)
    # lit_model.save_checkpoint()
    # lit_model = litmodel_class(args=args, tokenizer=tokenizer, num_relation=data.num_relation, num_entity = data.num_entity)
    # path = "output/epoch=1-Train/loss=0.92.ckpt"

    # if args.checkpoint:
    #     params_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    #     for k in list(params_dict.keys()):
    #         if "wte" in k:
    #             params_dict.pop(k)

    #     lit_model.load_state_dict(params_dict, strict=False)
    


    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="kgc", name=args.dataset)
        logger.log_hyperparams(vars(args))

    
    tester = pl.Trainer.from_argparse_args(args, default_root_dir="training/logs", gpus=args.gpus)
    #lit_model.load_checkpoint()
    result = tester.test(lit_model, data)


    # lit_model.load_state_dict(torch.load(path)["state_dict"])
    # print(path)

    # result = trainer.test(lit_model, data)
    print(result)





if __name__ == "__main__":

    main()