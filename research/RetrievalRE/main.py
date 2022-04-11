"""Experiment-running framework."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import time
import torch
import argparse
import importlib
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel
from pytorch_lightning.plugins import DDPPlugin


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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="DIALOGUE")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--best_model", type=str, default=None)
    parser.add_argument("--knn_lambda", type=float, default=0.2)
    parser.add_argument("--knn_topk", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

device = "cuda"
from tqdm import tqdm


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model)
    data_config = data.get_data_config()
    model.resize_token_embeddings(len(data.tokenizer))

    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    data.tokenizer.save_pretrained('test')

    logger = pl.loggers.TensorBoardLogger("training/logs")
    dataset_name = args.data_dir.split("/")[-1]
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")
        logger.log_hyperparams(vars(args))
    
    # init callbacks
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Eval/f1:.2f}',
        dirpath=args.output_dir,
        save_weights_only=True
    )
    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None


    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
        plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
    )

    if 'Embedding' not in args.litmodel_class:
        trainer.fit(lit_model, datamodule=data)

        # two steps

        path = model_checkpoint.best_model_path
        print(f"best model save path {path}")

        if not os.path.exists("config"):
            os.mkdir("config")
        config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
        day_name = time.strftime("%Y-%m-%d")
        if not os.path.exists(os.path.join("config", day_name)):
            os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))
        config = vars(args)
        config["path"] = path
        with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
            file.write(yaml.dump(config))

    if args.best_model:
        lit_model.load_state_dict(torch.load(args.best_model)["state_dict"])
        print("Load lit model successful!")

    
    if not args.two_steps: 
        if 'Get' in args.litmodel_class:
            data.setup()
            test_dataloader = DataLoader(data.data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            trainer.test(lit_model, test_dataloaders=test_dataloader)
        elif 'Combine' in args.litmodel_class:
            trainer.test(lit_model, datamodule=data)
        else:
            trainer.test()


if __name__ == "__main__":

    main()
