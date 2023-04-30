import os
import warnings
from argparse import ArgumentParser
from pprint import pprint
import random
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

import sys
sys.path.append(os.getcwd())
from src.models import model_mapping

warnings.filterwarnings("ignore")

def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--stable_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_alpha", type=float, default=1e-1)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--total_num_updates", type=int, default=200000)
    parser.add_argument("--warmup_updates", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="the name or the path to the pretrained model")
    parser.add_argument("--kge_model_type",
                        type=str,
                        default="KGEditor")
    parser.add_argument(
        "--ex_model_checkpoint",
        type=str,
        default="models/FT_KGE_E-FB15k237",
    )
    parser.add_argument("--data_type", type=str, default="FB15k237")
    parser.add_argument("--margin_kl_max", type=float, default=1e-1)
    parser.add_argument("--margin_kl_min", type=float, default=1e-3)
    parser.add_argument("--margin_lp_max", type=float, default=1e-6)
    parser.add_argument("--margin_lp_min", type=float, default=1e-9)
    parser.add_argument("--max_scale", type=float, default=1)
    parser.add_argument("--p", type=float, default=2)
    parser.add_argument("--divergences",
                        type=str,
                        choices=["kl", "lp", "both"],
                        default="kl")

    parser.add_argument("--optimizer",
                        type=str,
                        default="AdamW",
                        help="optimizer class from torch.optim")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bce", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--edit_num", type=int, default=4)
    parser.add_argument("--ex_size", type=int, default=64)
    parser.add_argument("--kb_layer", type=str, default="10,11")
    parser.add_argument(
        "--warm_up_radio",
        type=float,
        default=0.1,
        help="Number of examples to operate on per forward step.")

    return parser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dirpath",
                        type=str,
                        default="models/KGEditor/logger/FT_KGE")
    parser.add_argument("--save_top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="datasets/EditKnowledge_KG-BERT")
    parser.add_argument("--max_seq_length", type=int, default=64)
    
    parser.add_argument("--task_name", type=str, default="edit")

    parser = add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()

    seed_everything(seed=args.seed)

    logger = TensorBoardLogger(args.dirpath, name=None)

    callbacks = [
        ModelCheckpoint(
            monitor="Eval/hits3",
            mode="max",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=args.save_top_k,
            filename="model-{epoch:02d}-{Eval/hits3:.4f}-{valid_flipped:4f}",
        ),
        LearningRateMonitor(logging_interval="step", ),
    ]

    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=callbacks)

    model = model_mapping[args.kge_model_type](**vars(args))
        
    trainer.fit(model)
    # trainer.test(model)
    
