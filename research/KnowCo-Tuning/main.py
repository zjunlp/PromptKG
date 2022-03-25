import argparse
import importlib
from logging import debug

import numpy as np
import torch
import fcntl
import pytorch_lightning as pl
from transformers.file_utils import DATASETS_IMPORT_ERROR
import lit_models
import yaml
import time
from lit_models import TransformerLitModel
from transformers import AutoConfig
from utils import test_model
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--data_class", type=str, default="KGC")
    parser.add_argument("--chunk", type=str, default="")
    parser.add_argument("--model_class", type=str, default="RobertaUseLabelWord")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser



def main():
    parser = _setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    data = data_class(args)
    tokenizer = data.tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)


    

    # only in bertuselabelword
    if args.prompt and "Use" in args.model_class:
        with open(os.path.join(args.data_dir, "label_word.txt"), "r") as file:
            label_word = []
            for line in file.readlines():
                label_word.append(tokenizer(line.replace("\n",""), add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))
            model.init_unused_weights(label_word)



    lit_model = litmodel_class(args=args, model=model)
    


    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="kgc")
        logger.log_hyperparams(vars(args))

    if args.load_checkpoint:
        lit_model.load_state_dict(torch.load(args.load_checkpoint)['state_dict'])
        import IPython; IPython.embed(); exit(1)
        test_model(args, lit_model.model, data.get_tokenizer(), logger)

        return


    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/acc", mode="max", patience=10)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/acc", mode="max",
        filename='{epoch}-{Eval/acc:.2f}',
        dirpath="output",
        save_weights_only=True
    )


    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs")


    trainer.fit(lit_model, datamodule=data)

    path = model_checkpoint.best_model_path

    lit_model.load_state_dict(torch.load(path)["state_dict"])
    print(path)

    # result = trainer.test()

    test_model(args, lit_model.model, data.get_tokenizer(), logger)




if __name__ == "__main__":

    main()
