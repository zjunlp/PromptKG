import math
import os
import logging
import numpy as np
from copy import deepcopy
from higher.patch import monkeypatch as make_functional
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from src.models.modeling_bert import BertForMaskedLM, BertEncoder, EXBertForMaskedLM
from src.data.data_module import dataset_mapping

class Args:
  def __init__(self, **entries) -> None:
      self.__dict__.update(entries)

logger = logging.getLogger(__name__)

class Adapter(nn.Module):
    def __init__(self, args, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)
        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

class PretrainedModel(nn.Module):
    def __init__(self, model_name_or_path, kge_model_type):
        super(PretrainedModel, self).__init__()
        pretrain_model = BertForMaskedLM.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.model = pretrain_model.bert
        self.cls = pretrain_model.cls
        self.config = pretrain_model.config
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions) bert: logits, (hidden_states), (attetions)

class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config, n_rel):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = self.args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float= 0.1
            hidden_dropout_prob: float=0.1
            hidden_size: int=768
            initializer_range: float=0.02
            intermediate_size: int=3072
            layer_norm_eps: float=1e-05
            max_position_embeddings: int=514
            num_attention_heads: int=12
            num_hidden_layers: int=self.args.adapter_transformer_layers
            num_labels: int=2
            output_attentions: bool=False
            output_hidden_states: bool=False
            torchscript: bool=False
            type_vocab_size: int=1
            vocab_size: int=50265
            chunk_size_feed_forward: int=self.config.chunk_size_feed_forward
            add_cross_attention: bool=self.config.add_cross_attention

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.num_labels = n_rel
        # self.config.output_hidden_states=True
        self.adapter_list = args.adapter_list
        # self.adapter_list =[int(i) for i in self.adapter_list]
        self.adapter_num = len(self.adapter_list)
        # self.adapter = Adapter(args, AdapterConfig)

        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

        self.com_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.out_proj = None

    def forward(self, pretrained_model_outputs, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        outputs = pretrained_model_outputs
        sequence_output = outputs[0] # last hidden states
        # pooler_output = outputs[1]
        hidden_states = outputs[1] # (8, 64, 768)
        num = len(hidden_states)

        hidden_states_last = torch.zeros(sequence_output.size()).to(self.args.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1: # if adapter_skip_layers>=1, skip connection
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]

        ##### drop below parameters when doing downstream tasks
        com_features = self.com_dense(torch.cat([sequence_output, hidden_states_last],dim=2))

        # subj_special_start_id = subj_special_start_id.unsqueeze(1)
        # subj_output = torch.bmm(subj_special_start_id, com_features)
        # obj_special_start_id = obj_special_start_id.unsqueeze(1)
        # obj_output = torch.bmm(obj_special_start_id, com_features)
        
        pos = (input_ids == self.config.mask_token_id).nonzero(as_tuple=True)
        
        logits = self.out_proj(
            self.dropout(self.dense(com_features)))
        
        logits = logits[pos[0], pos[1], self.config.entity_id_st:self.config.entity_id_ed]
        
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)

class ConditionedParameter(torch.nn.Module):
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape

        if len(self.parameter_shape) == 2: # condition_dim是从lstm中得到的tensor，然后用linear学习返回到768作为更新的parm_dict
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):

        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb

        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split(
                [self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1
            )
        else:
            raise RuntimeError()

        return (
            self.max_scale
            * torch.mean(conditioner_norm.sigmoid(), dim=0).squeeze() # 多条我们直接取mean
            * (grad * a.squeeze() + b.squeeze())
        )


class LSTMConditioner(torch.nn.Module):
    def __init__(
        self,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=256,
        output_dim=1024,
        embedding_init=None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_init,
        )
        self.lstm = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = FeedForward(
            input_dim=hidden_dim * 2,
            num_layers=1,
            hidden_dims=[output_dim],
            activations=[torch.nn.Tanh()],
        )

    def forward(self, inputs, masks):
        return self.linear(self.lstm(self.embedding(inputs), masks)) # 1, 64


class OneShotLearner(torch.nn.Module):
    def __init__(
        self,
        model,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=128,
        condition_dim=1024,
        include_set={},
        max_scale=1e-3,
        embedding_init=None,
    ):
        super().__init__()

        self.param2conditioner_map = {
            n: "{}_conditioner".format(n).replace(".", "_")
            for n, p in model.named_parameters()
            if n in include_set
        }

        self.conditioners = torch.nn.ModuleDict(
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    condition_dim,
                    hidden_dim,
                    max_scale=max_scale,
                )
                for n, p in model.named_parameters()
                if n in include_set
            }
        )

        self.condition = LSTMConditioner(
            vocab_dim,
            embedding_dim,
            hidden_dim,
            condition_dim,
            embedding_init=embedding_init,
        )

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks) # LSTM输出condition
        return {
            p: self.conditioners[self.param2conditioner_map[p]](
                condition,
                grad=grads[p] if grads else None,
            )
            for p, c in self.param2conditioner_map.items()
        }

class BaseKEModule(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.edit_num,
            collate_fn=self.sampler,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=True
        )
    
    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.edit_num,
            collate_fn=self.sampler,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=True
        )

    def get_logits_orig_params_dict(self, batch):

        with torch.enable_grad():
            logits = self.ex_model.eval()(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            
            logits_orig, _ = logits.split([
                len(batch["input_ids"]) - self.edit_num,
                self.edit_num,
            ])
            input_ids = batch['input_ids']
            batch_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[batch_idx, mask_idx, self.entity_id_st:self.entity_id_ed]
            
            #  get info of gradient
            grads = torch.autograd.grad(
                # cross_entropy
                torch.nn.functional.cross_entropy(
                    mask_logits[-self.edit_num:, :],
                    batch["label"][-self.edit_num:],
                    reduction="none",
                ).mean(-1),
                self.ex_model.parameters(),
            )

        grads = {
            name: grad
            for (name, _), grad in zip(self.ex_model.named_parameters(), grads)
        }

        params_dict = self.ex_learner(
            batch["cond_input_ids"][-self.edit_num:],
            batch["cond_attention_mask"][-self.edit_num:],
            grads=grads,
        )

        return logits_orig.detach()[:, :, self.entity_id_st:self.entity_id_ed], params_dict

    def forward(self, batch, logits_orig=None, params_dict=None):

        if not params_dict:
            logits_orig, params_dict = self.get_logits_orig_params_dict(batch)

        fmodel = make_functional(self.ex_model).eval()

        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.ex_model.named_parameters()
            ],
        ).logits
        
        return logits_orig, logits[:, :, self.entity_id_st:self.entity_id_ed], params_dict
    
    def get_kl_lp_cr(self, logits_orig, logits, label, params_dict, input_ids):
        # Reliability
        pos = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        # 
        kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(torch.nn.functional.softmax(logits_orig[pos[0][:-self.edit_num], pos[1][:-self.edit_num], :])),
            torch.distributions.Categorical(
               torch.nn.functional.softmax(logits[pos[0][:-self.edit_num], pos[1][:-self.edit_num], :])),
        )
        
        # don't update too much params
        lp = sum((p.abs()**self.hparams.p).mean()**(1 / self.hparams.p)
                 for p in params_dict.values()) / len(params_dict)

        # ensure the result which has been edited
        cr = torch.nn.functional.cross_entropy(
            logits[pos[0][-self.edit_num:], pos[1][-self.edit_num:], :],
            label[-self.edit_num:],
            reduction="none",
        ).mean(-1)

        return kl, lp, cr

    def training_step(self, batch, batch_idx=None):

        logits_orig, logits, params_dict = self.forward(batch)

        kl, lp, cr = self.get_kl_lp_cr(logits_orig, logits, batch['label'],
                                       params_dict, batch["input_ids"])
        kl = kl.mean(-1) 

        loss_kl = self.alpha_kl * (kl - self.margin_kl) # margin_kl is too large
        loss_lp = self.alpha_lp * (lp - self.margin_lp)

        if self.hparams.divergences == "both":
            loss = cr + loss_kl + loss_lp
        elif self.hparams.divergences == "kl":
            loss = cr + loss_kl
        elif self.hparams.divergences == "lp":
            loss = cr + loss_lp

        self.log("alpha_kl",
                 self.alpha_kl,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)
        self.log("alpha_lp",
                 self.alpha_lp,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)
        self.log("kl", kl, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lp", lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("cr", cr, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": torch.abs(loss)}

    def _eval(self, batch, batch_idx, ):
        logits_orig, params_dict = self.get_logits_orig_params_dict(batch)
        fmodel = make_functional(self.ex_model).eval()
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')
        my_keys = list(batch.keys())
        for k in my_keys:
            if k not in ["input_ids", "attention_mask", "token_type_ids"]:
                batch.pop(k)    
        
        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.ex_model.named_parameters()
            ],
        ).logits[:, :, self.entity_id_st:self.entity_id_ed]
        
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx]

        _, outputs = torch.sort(logits, dim=1, descending=True) # outputs代表
        edit_entity_order = outputs[-self.edit_num:, :]
        edit_input_ids = batch["input_ids"][-self.edit_num:, :]
        edit_labels = label[-self.edit_num:]
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        edit_ranks = ranks[-self.edit_num:]
        
        return dict(ranks = np.array(ranks), edit_entity_order = np.array(edit_entity_order.detach().cpu()), edit_input_ids = edit_input_ids, edit_labels = edit_labels, edit_ranks = edit_ranks)

    def validation_step(self, batch, batch_idx=None):

        result = self._eval(batch, batch_idx)
        return result
    
    def validation_epoch_end(self, outputs):
        if self.epoch % 5 == 0:
          output_dir = f"models/{self.hparams.kge_model_type}/output/{str(self.hparams.data_type)}/{str(self.task_name)}/"
          if not os.path.exists(output_dir):
              os.makedirs(output_dir)
          torch.save(self.ex_learner, os.path.join(output_dir, f"{str(self.epoch)}_params.pt"))

        self.epoch = self.epoch + 1
        
        loc_ranks = np.concatenate([_['ranks'][:-self.edit_num] for _ in outputs])
        edit_ranks = np.concatenate([_['ranks'][-self.edit_num:] for _ in outputs])

        loc_hits5 = (loc_ranks<=5).mean()
        loc_hits3 = (loc_ranks<=3).mean()
        loc_hits1 = (loc_ranks<=1).mean()
        edit_hits5 = (edit_ranks<=5).mean()
        edit_hits3 = (edit_ranks<=3).mean()
        edit_hits1 = (edit_ranks<=1).mean()
        print("Eval/ranks", edit_ranks)
        print("Eval/hits5", edit_hits5)
        print("Eval/hits3", edit_hits3)
        print("Eval/hits1", edit_hits1)
        print("Eval/mean_rank", edit_ranks.mean())
        print("Eval/mrr", (1. / edit_ranks).mean())
        print("Eval/loc_hits1", loc_hits1)
        print("Eval/loc_hits3", loc_hits3)
        print("Eval/loc_hits5", loc_hits5)
        print("Eval/loc_mean_rank", loc_ranks.mean())
        print("Eval/loc_mrr", (1. / loc_ranks).mean())

        self.log("Eval/loc_hits1", loc_hits1)
        self.log("Eval/loc_hits3", loc_hits3)
        self.log("Eval/loc_hits5", loc_hits5)
        self.log("Eval/loc_mean_rank", loc_ranks.mean())
        self.log("Eval/loc_mrr", (1. / loc_ranks).mean())
        self.log("Eval/hits5", edit_hits5)
        self.log("Eval/hits3", edit_hits3)
        self.log("Eval/hits1", edit_hits1)
        self.log("Eval/mean_rank", edit_ranks.mean())
        self.log("Eval/mrr", (1. / edit_ranks).mean())
        
        return super().validation_epoch_end(outputs)
        

    def sample(
        self,
        sentences,
        condition,
        logits_orig=None,
        params_dict=None,
        stop_condition=None,
    ):
        len_sent = len(sentences)
        with torch.no_grad():
            logits_orig, logits, params_dict = self.forward(
                {
                    k: v.to(self.device)
                    for k, v in self.val_dataset.get_batch(
                        sentences, condition).items()
                },
                logits_orig=logits_orig,
                params_dict=params_dict,
            )

            n_iter = 1
            if stop_condition is not None and stop_condition(
                    condition, logits, n_iter):
                model_tmp = deepcopy(self.model)
                params_dict_tmp = deepcopy(params_dict)

                while stop_condition(condition, logits, n_iter):
                    for n, p in self.model.named_parameters():
                        p.data += params_dict.get(n, 0)

                    _, logits, params_dict = self.forward({
                        k: v.to(self.device)
                        for k, v in self.val_dataset.get_batch(
                            sentences, condition).items()
                    })
                    params_dict_tmp = {
                        k: v + params_dict[k]
                        for k, v in params_dict_tmp.items()
                    }
                    n_iter += 1

                self.model = model_tmp
                params_dict = params_dict_tmp

            return logits_orig, logits[:len_sent], params_dict

    def on_before_zero_grad(self, optimizer):
        self.alpha_kl.data = torch.where(
            self.alpha_kl.data < 0,
            torch.full_like(self.alpha_kl.data, 0),
            self.alpha_kl.data,
        )
        self.alpha_lp.data = torch.where(
            self.alpha_lp.data < 0,
            torch.full_like(self.alpha_lp.data, 0),
            self.alpha_lp.data,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            [
                {
                    "params": self.ex_learner.parameters(),
                    "lr": self.hparams.lr,
                },
                {
                    "params": [self.alpha_kl, self.alpha_lp],
                    "lr": self.hparams.lr_alpha,
                },
            ],
            centered=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }]

class KGEditor(BaseKEModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        config = AutoConfig.from_pretrained(
            self.hparams.ex_model_checkpoint,
        )
        
        config.ex_size = self.hparams.ex_size
        config.kb_layer = [int(i) for i in self.hparams.kb_layer.split(',') if i != '']
        
        self.ex_model = EXBertForMaskedLM.from_pretrained(
            self.hparams.ex_model_checkpoint, config=config).eval()
        
        data = dataset_mapping[self.hparams.kge_model_type](Args(**kwargs))
        data.setup()

        self.__dict__.update(data.get_config())
        
        self.task_name = self.hparams.task_name
        
        self.epoch = 0
        self.edit_num = self.hparams.edit_num
        
        self.ex_learner = OneShotLearner(
            self.ex_model,
            vocab_dim=self.ex_model.bert.embeddings.word_embeddings.weight.data.
            shape[0],
            embedding_dim=self.ex_model.bert.embeddings.word_embeddings.weight.
            data.shape[1],
            hidden_dim=128,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.ex_model.named_parameters()
                if ("dense_in_ex" in n.lower() or "dense_out_ex" in n.lower()) and "bias" not in n.lower()
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.ex_model.bert.embeddings.word_embeddings.weight.
            data,
        )
        
        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl",
                             torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp",
                             torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []

class KnowledgeEditor(BaseKEModule):
  
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForMaskedLM.from_pretrained(
            self.hparams.model_checkpoint).eval()

        self.task_name = self.hparams.task_name
        
        data = dataset_mapping[self.hparams.kge_model_type](Args(**kwargs))
        data.setup()
        self.__dict__.update(data.get_config())
        
        self.epoch = 0
        self.edit_num = self.hparams.edit_num
        
        self.learner = OneShotLearner(
            self.model,
            vocab_dim=self.model.bert.embeddings.word_embeddings.weight.data.
            shape[0],
            embedding_dim=self.model.bert.embeddings.word_embeddings.weight.
            data.shape[1],
            hidden_dim=128,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.model.named_parameters()
                if all(e not in n.lower() for e in (
                    "bias",
                    "norm",
                    "embeddings",
                    "cls",
                    "pooler",
                    "shared",
                    "embed",
                    "positions",
                ))
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.model.bert.embeddings.word_embeddings.weight.
            data,
        )

        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl",
                             torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp",
                             torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []
    
    def get_logits_orig_params_dict(self, batch):

        with torch.enable_grad():
            logits = self.model.eval()(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            
            logits_orig, _ = logits.split([
                len(batch["input_ids"]) - self.edit_num,
                self.edit_num,
            ])
            input_ids = batch['input_ids']
            batch_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[batch_idx, mask_idx, self.entity_id_st:self.entity_id_ed]
            
            #  get info of gradient
            grads = torch.autograd.grad(
                # cross_entropy
                torch.nn.functional.cross_entropy(
                    mask_logits[-self.edit_num:, :],
                    batch["label"][-self.edit_num:],
                    reduction="none",
                ).mean(-1),
                self.model.parameters(),
            )

        grads = {
            name: grad
            for (name, _), grad in zip(self.model.named_parameters(), grads)
        }

        params_dict = self.learner(
            batch["cond_input_ids"][-self.edit_num:],
            batch["cond_attention_mask"][-self.edit_num:],
            grads=grads,
        )

        return logits_orig.detach()[:, :, self.entity_id_st:self.entity_id_ed], params_dict

    def forward(self, batch, logits_orig=None, params_dict=None):

        if not params_dict:
            logits_orig, params_dict = self.get_logits_orig_params_dict(batch)

        fmodel = make_functional(self.model).eval()

        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.model.named_parameters()
            ],
        ).logits
        
        return logits_orig, logits[:, :, self.entity_id_st:self.entity_id_ed], params_dict
    
    def _eval(self, batch, batch_idx, ):
        logits_orig, params_dict = self.get_logits_orig_params_dict(batch)
        fmodel = make_functional(self.model).eval()
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')
        my_keys = list(batch.keys())
        for k in my_keys:
            if k not in ["input_ids", "attention_mask", "token_type_ids"]:
                batch.pop(k)    
        
        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.model.named_parameters()
            ],
        ).logits[:, :, self.entity_id_st:self.entity_id_ed]
        
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx]

        _, outputs = torch.sort(logits, dim=1, descending=True) # outputs代表
        edit_entity_order = outputs[-self.edit_num:, :]
        edit_input_ids = batch["input_ids"][-self.edit_num:, :]
        edit_labels = label[-self.edit_num:]
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        edit_ranks = ranks[-self.edit_num:]
        
        return dict(ranks = np.array(ranks), edit_entity_order = np.array(edit_entity_order.detach().cpu()), edit_input_ids = edit_input_ids, edit_labels = edit_labels, edit_ranks = edit_ranks)

    def validation_step(self, batch, batch_idx=None):

        result = self._eval(batch, batch_idx)
        return result
    
    def validation_epoch_end(self, outputs):
        if self.epoch % 5 == 0:
          output_dir = f"models/{self.hparams.kge_model_type}/output/{str(self.hparams.data_type)}/{str(self.task_name)}/"
          if not os.path.exists(output_dir):
              os.makedirs(output_dir)
          torch.save(self.learner, os.path.join(output_dir, f"{str(self.epoch)}_params.pt"))

        self.epoch = self.epoch + 1
        
        loc_ranks = np.concatenate([_['ranks'][:-self.edit_num] for _ in outputs])
        edit_ranks = np.concatenate([_['ranks'][-self.edit_num:] for _ in outputs])

        loc_hits5 = (loc_ranks<=5).mean()
        loc_hits3 = (loc_ranks<=3).mean()
        loc_hits1 = (loc_ranks<=1).mean()
        edit_hits5 = (edit_ranks<=5).mean()
        edit_hits3 = (edit_ranks<=3).mean()
        edit_hits1 = (edit_ranks<=1).mean()
        print("Eval/ranks", edit_ranks)
        print("Eval/hits5", edit_hits5)
        print("Eval/hits3", edit_hits3)
        print("Eval/hits1", edit_hits1)
        print("Eval/mean_rank", edit_ranks.mean())
        print("Eval/mrr", (1. / edit_ranks).mean())
        print("Eval/loc_hits1", loc_hits1)
        print("Eval/loc_hits3", loc_hits3)
        print("Eval/loc_hits5", loc_hits5)
        print("Eval/loc_mean_rank", loc_ranks.mean())
        print("Eval/loc_mrr", (1. / loc_ranks).mean())

        self.log("Eval/loc_hits1", loc_hits1)
        self.log("Eval/loc_hits3", loc_hits3)
        self.log("Eval/loc_hits5", loc_hits5)
        self.log("Eval/loc_mean_rank", loc_ranks.mean())
        self.log("Eval/loc_mrr", (1. / loc_ranks).mean())
        self.log("Eval/hits5", edit_hits5)
        self.log("Eval/hits3", edit_hits3)
        self.log("Eval/hits1", edit_hits1)
        self.log("Eval/mean_rank", edit_ranks.mean())
        self.log("Eval/mrr", (1. / edit_ranks).mean())
        
        return super().validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            [
                {
                    "params": self.learner.parameters(),
                    "lr": self.hparams.lr,
                },
                {
                    "params": [self.alpha_kl, self.alpha_lp],
                    "lr": self.hparams.lr_alpha,
                },
            ],
            centered=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }]

model_mapping = {
    'KGEditor': KGEditor,
    'KE': KnowledgeEditor,
}