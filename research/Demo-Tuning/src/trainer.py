import gc
import torch
import logging
from tqdm import trange, tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.optim import optimizer
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import TrainOutput


logger = logging.getLogger(__name__)


def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]
 
    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
    
    def train(self, model_path=None, dev_objective=None):
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        train_dataloader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + \
                                int(self.args.max_steps % num_update_steps_per_epoch > 0)
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        model = self.model

        # without regard to fp16, tpu or multi-gpu
        # in view of 'gradient_accumulation'
        total_train_batch_size = (self.args.train_batch_size * self.args.gradient_accumulation_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch"
        )
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            for step, inputs in enumerate(epoch_iterator):
                tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs['norm'] = norm.item()
                        logs['learning_rate'] = scheduler.get_last_lr()[0]
                        logging_loss_scalar = tr_loss_scalar
                        
                        self.log(logs)
                
                    metrics = None
                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        output = self.evaluate()
                        metrics = output.metrics
                        objective = self.dev_objective(metrics)
                        if objective > self.objective:
                            logger.info("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir)
                    
                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        return TrainOutput(self.global_step, tr_loss / self.global_step, {'metric': self.objective})

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        
        return output

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def training_step(self, model: nn.Module, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if len(inputs) == 2:
            loss_weight = None

            loss, cl_loss = self.compute_loss(model, inputs[0], inputs[1], return_seq_output=True)
            cl_loss = -2 * torch.sum(cl_loss * loss_weight) if loss_weight is not None else -2 * torch.mean(cl_loss)
            loss = loss + cl_loss * self.args.lambda_cl

        else:
            loss = self.compute_loss(model, inputs, return_seq_output=True)

        if self.args.gradient_accumulation_steps > 1:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def compute_byol_cl_loss(self, target_model, inputs2, outputs):
        outputs2 = target_model(**inputs2, return_output=True, only_mask_output=True)

        v1 = outputs[2]     # [bs, hs]
        v2 = outputs2[0]    # [bs, hs]

        norm1 = v1.norm(dim=-1)           # [bs]
        norm2 = v2.norm(dim=-1)           # [bs]

        # operate `mean` in `training_step` due to the requirement of weighted CL.
        cl_loss = (torch.sum(v1 * v2, dim=-1, keepdim=True) / (norm1 * norm2)).squeeze(-1)        # [bs, 1]

        return cl_loss, outputs2[0]

    def compute_loss(self, model, inputs, inputs2=None, return_outputs=False, return_seq_output=False, target_model=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs, return_output=return_seq_output)

        if inputs2 is not None:
            cl_loss, _ = self.compute_byol_cl_loss(model, inputs2, outputs)
            
        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if inputs2 is not None:
            loss = (loss, cl_loss)

        return (loss, outputs) if return_outputs else loss
    

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        if 'v2' in inputs:
            inputs1 = inputs['v1']
            inputs2 = inputs['v2']
        else:
            inputs1 = inputs
            inputs2 = None

        for k, v in inputs1.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                inputs1[k] = v.to(**kwargs)

        if inputs2 is not None:
            for k, v in inputs2.items():
                if isinstance(v, torch.Tensor):
                    kwargs = dict(device=self.args.device)
                    inputs2[k] = v.to(**kwargs)
        
        if inputs2 is not None:
            return inputs1, inputs2
        else:
            return inputs1