import json
import numpy as np

def rank_score(ranks):
	# prepare the dataset
	len_samples = len(ranks)
	hits10 = [0] * len_samples
	hits5 = [0] * len_samples
	hits1 = [0] * len_samples
	mrr = []


	for idx, rank in enumerate(ranks):
		if rank <= 10:
			hits10[idx] = 1.
			if rank <= 5:
				hits5[idx] = 1.
				if rank <= 1:
					hits1[idx] = 1.
		mrr.append(1./rank)
	

	return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

def acc(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return (preds == labels).mean()
import torch.nn as nn
import torch
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SparseMax(nn.Module):
    """Sparsemax loss"""

    def __init__(self, k_sparse=1):
        super(SparseMax, self).__init__()
        self.k_sparse = k_sparse
        
    def forward(self, preds, labels):
        """
        Args:
            preds (torch.Tensor):  [batch_size, number_of_logits]
            labels (torch.Tensor): [batch_size] index, not ont-hot
        Returns:
            torch.Tensor
        """
        preds = preds.reshape(preds.size(0), -1) # [batch_size, -1]
        topk = preds.topk(self.k_sparse, dim=1)[0] # [batch_size, k_sparse]
        
        # log(sum(exp(topk)))
        pos_loss = torch.logsumexp(topk, dim=1)
        # s_t
        neg_loss = torch.gather(preds, 1, labels[:, None].expand(-1, preds.size(1)))[:, 0]
        
        return (pos_loss - neg_loss).mean()
    
class SparseMax_good(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super().__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


from copy import deepcopy
from typing import Optional, Union, Dict, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only


class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """
    def __init__(self, decay = 0.9999,
        ema_device = None,
        pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.
        
        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()
        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> dict:
        return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]

class LAMA_metrics():
    def __init__(self) -> None:
        pass

    def get_ranking(self, log_probs, masked_indices, vocab, label_index = None, index_list = None, topk = 1000, P_AT = 10, print_generation=False):

        experiment_result = {}

        log_probs, index_max_probs, value_max_probs = self.__max_probs_values_indices(masked_indices, log_probs, topk=topk)
        result_masked_topk = self.__print_top_k(value_max_probs, index_max_probs, vocab, topk, index_list)
        experiment_result['topk'] = result_masked_topk

        MRR = 0.
        P_AT_X = 0.
        P_AT_1 = 0.
        PERPLEXITY = None

        if label_index is not None:

            # check if the labe_index should be converted to the vocab subset
            if index_list is not None:
                label_index = index_list.index(label_index)

            query = torch.full(value_max_probs.shape, label_index, dtype=torch.long).numpy().astype(int)
            ranking_position = (index_max_probs==query).nonzero()

            # LABEL PERPLEXITY
            tokens = torch.from_numpy(np.asarray(label_index))
            label_perplexity = log_probs.gather(
                dim=0,
                index=tokens,
            )
            PERPLEXITY = label_perplexity.item()

            if len(ranking_position) >0 and ranking_position[0].shape[0] != 0:
                rank = ranking_position[0][0] + 1

                # print("rank: {}".format(rank))

                if rank >= 0:
                    MRR = (1/rank)
                if rank >= 0 and rank <= P_AT:
                    P_AT_X = 1.
                if rank == 1:
                    P_AT_1 = 1.

        experiment_result["MRR"] = MRR
        experiment_result["P_AT_X"] = P_AT_X
        experiment_result["P_AT_1"] = P_AT_1
        experiment_result["PERPLEXITY"] = PERPLEXITY
        #
        # print("MRR: {}".format(experiment_result["MRR"]))
        # print("P_AT_X: {}".format(experiment_result["P_AT_X"]))
        # print("P_AT_1: {}".format(experiment_result["P_AT_1"]))
        # print("PERPLEXITY: {}".format(experiment_result["PERPLEXITY"]))

        return experiment_result

    @staticmethod
    def __max_probs_values_indices(masked_indices, log_probs, topk=1000):

        # score only first mask
        masked_indices = masked_indices[:1]

        masked_index = masked_indices[0]
        log_probs = log_probs[masked_index]

        value_max_probs, index_max_probs = torch.topk(input=log_probs,k=topk,dim=0)
        index_max_probs = index_max_probs.numpy().astype(int)
        value_max_probs = value_max_probs.detach().numpy()

        return log_probs, index_max_probs, value_max_probs

    @staticmethod
    def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts = 10):
        result = []
        for i in range(mask_topk):
            filtered_idx = index_max_probs[i].item()

            if index_list is not None:
                # the softmax layer has been filtered using the vocab_subset
                # the original idx should be retrieved
                idx = index_list[filtered_idx]
            else:
                idx = filtered_idx

            log_prob = value_max_probs[i].item()
            word_form = vocab[idx]
            
            element = {'i' : i, 'token_idx': idx, 'log_prob': log_prob, 'token_word_form': word_form}
            result.append(element)
        return result

class LAMATester(pl.Trainer):
    def __init__(self) -> None:
        super().__init__()

    def test(
        self,
        model = None,
        dataloaders = None,
        ckpt_path = None,
        verbose = True,
        datamodule = None,
    ):
        r"""
        Perform one evaluation epoch over the test set.
        It's separated from fit to make sure you never run on your test set until you want to.

        Args:
            model: The model to test.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying test samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the test results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the test phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`,
            :meth:`~pytorch_lightning.core.lightning.LightningModule.test_epoch_end`, etc.
            The length of the list corresponds to the number of test dataloaders used.
        """
        self.strategy.model = model or self.lightning_module
        return self._call_and_handle_interrupt(self._test_impl, model, dataloaders, ckpt_path, verbose, datamodule)