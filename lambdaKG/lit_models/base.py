import argparse
import pytorch_lightning as pl
import torch
from typing import Dict, Any

from transformers.optimization import get_linear_schedule_with_warmup


OPTIMIZER = "AdamW"
LR = 5e-5
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100
import os

class MyTrainer(pl.Trainer):
    def save_checkpoint(self, filepath, weights_only=False):
        if self.is_global_zero:
            dirpath = os.path.split(filepath)[0]
            lightningmodel = self.get_model()
            assert hasattr(lightningmodel.model, "save_pretrained"), "Use huggingface save pretrained method."
            lightningmodel.model.save_pretrained(dirpath)
            lightningmodel.tokenizer.save_pretrained(dirpath)
            lightningmodel.model.config.save_pretrained(dirpath)

class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, args):
        super().__init__()
        self.args = Config(vars(args)) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)
        self.checkpoint = self.args.get("checkpoint")

        
        self._init_model()

    def _init_model(self,):
        return None

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER,
                            help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--ema_decay", type=float, default=0.0)
        return parser

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False,
                 on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (
            dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps is not None and self.trainer.max_steps != -1 and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(
            optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
    
    def save_checkpoint(self, path=None) -> None:
        path = self.checkpoint if path == None else path
        if path == None:
            return
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.model.state_dict(), path)
        print(f"Save model at {path}.")

    def load_checkpoint(self, path=None) -> None:
        path = self.checkpoint if path == None else path
        if path == None:
            raise RuntimeError("Load model failed: model path is invalid.")
        self.model.load_state_dict(torch.load(path))
        print(f"Load model from {path}.")
        

