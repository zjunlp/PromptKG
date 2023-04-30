import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import transformers
import higher
import logging
from higher.patch import monkeypatch as make_functional
from collections import defaultdict

from editable_model import EditableModel
from hooks import hook_model
import nn as local_nn
from utils import _logits, _inner_params

LOG = logging.getLogger(__name__)


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)

    return new_m, new_s


class GradientTransform(nn.Module): # define a IDMLP which input_dim is x_dim + delta_dim (combine)
    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes = None):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.cfg = cfg
        if cfg.combine and (cfg.one_sided or cfg.x_only or cfg.delta_only):
            raise ValueError("cfg.combine cannot be used with one-sided MEND variants")

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = getattr(local_nn, cfg.mlp_class)
        LOG.info(f"Building Gradient Transform with MLP class {MlpClass}")

        def delta_net():
            return MlpClass(delta_dim, delta_dim, delta_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def x_net():
            return MlpClass(x_dim, x_dim, x_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def combined_net():
            return MlpClass(delta_dim + x_dim, delta_dim + x_dim, (delta_dim + x_dim) * 2,
                            cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def ID():
            return lambda x, mode=None: x

        if cfg.combine:
            self.mlp = combined_net()
        elif cfg.one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])

        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]

        if self.training:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(u_[idx], self.u_mean, self.u_s, self.k)
                    self.v_mean, self.v_s = update_counter(v_[idx], self.v_mean, self.v_s, self.k)

            if self.k < 2:
                raise RuntimeError(f"Can't perform normalization with only {self.k} samples so far")
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.cfg.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_

        if self.cfg.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(v_input, mode=param_idx)


class MEND(EditableModel):
    def get_shape(self, p):
        # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
        return p.shape if isinstance(self.model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

    # model_constructor
    def __init__(self, model, config, data, model_constructor, mend=None, edit_lrs=None):
        super().__init__(model, config, model_constructor)
        
        self.data = data
        if config.model.from_pretrain.split('/')[-1] == "LMKE":
            self.model_name = "LMKE"
        else:
            self.model_name = "KNN-KG"

        if edit_lrs is None:
            edit_lrs = nn.Parameter(torch.tensor([config.edit_lr] * len(self.config.model.inner_params)))
        self.edit_lrs = edit_lrs

        if not hasattr(self.model, "handles"):
            hook_model(self.model, self.config.model.inner_params)
            LOG.info(f"Hooked {len(self.model.handles)//2} modules")

        if config.mend.shared:
            shape_dict = defaultdict(list)
            for n, p in _inner_params(model.named_parameters(), self.config.model.inner_params):
                shape_dict[self.get_shape(p)].append(n) # params mapping
            self.shape_dict = shape_dict

        if mend is None:
            if not config.mend.shared:
                self.mend = nn.ModuleDict({
                    n.replace(".", "#"): GradientTransform(*self.get_shape(p), config.mend)
                    for (n, p) in _inner_params(model.named_parameters(), self.config.model.inner_params)
                })
            else:
                self.mend = nn.ModuleDict({
                    str(tuple(s)): GradientTransform(*s, config.mend, len(shape_dict[s]))
                    for s in shape_dict.keys()
                })
        else:
            self.mend = mend

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        model_keys = self.model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert len([k for k in res.missing_keys if not k.startswith("model.")]) == 0, "Should only have missing keys for model."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self):
        return list(self.mend.parameters()) + [self.edit_lrs]

    def edit(self, batch, labels=None, condition=None, detach_history=False):
        outputs = _logits(self.model(**batch))[:, :, self.data.entity_id_st:self.data.entity_id_ed] # 获得classifier的output
        _, mask_idx = (batch['input_ids'] == self.data.tokenizer.mask_token_id).nonzero(as_tuple=True)
        outputs = outputs[torch.arange(batch['input_ids'].shape[0]), mask_idx]
        loss = self.edit_loss_fn(outputs, labels)["nll"] # edit的loss, 更改后的label

        names = set([n for n, p in self.model.named_parameters()])
        pset = set(self.config.model.inner_params)
        for p in pset: # 可去掉加速
            assert p in names, f"inner param {p} not in model"

        loss.backward() # loss回传得到梯度

        if self.config.mend.shared:
            param_idx = lambda n, p: self.shape_dict[self.get_shape(p)].index(n) if self.config.mend.shared else None  # noqa: E731
            transformed_factors = {
                n: self.mend[str(tuple(self.get_shape(p)))](p.__x__, p.__delta__, param_idx(n, p)) # __x__ (16, 64, 768)当前层最后的输出 __delta__ (3072, 768)是grad梯度
                for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params)
            }
        else:
            transformed_factors = {
                n: self.mend[n.replace(".", "#")](p.__x__, p.__delta__) # self.mend是一个MLP
                for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params)
            }

        # Should be bi,bj->ji for nn.Linear, but GPT2 uses Conv1d instead...
        if isinstance(self.model, transformers.GPT2LMHeadModel):
            targ = "ij"
        else:
            targ = "ji"
        mean_grads = { # 更新的梯度
            n: torch.einsum(f"bi,bj->{targ}", x, delta)
            for n, (x, delta) in transformed_factors.items() # transformed_factors是mlp输出的结果
        }

        info_dict = {}
        idx = 0
        for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params):
            info_dict[f"grad/true_mag{idx}"] = p.grad.norm(2).item()
            info_dict[f"grad/pseudo_mag{idx}"] = mean_grads[n].norm(2).item()
            info_dict[f"grad/true_std{idx}"] = p.grad.std().item()
            info_dict[f"grad/pseudo_std{idx}"] = mean_grads[n].std().item()
            info_dict[f"grad/diff{idx}"] = (p.grad - mean_grads[n]).norm(2).item()
            info_dict[f"grad/cos{idx}"] = F.cosine_similarity(p.grad.reshape(-1), mean_grads[n].reshape(-1), dim=0).item()
            idx += 1

        self.model.zero_grad()

        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        updates = {n: lr * g for lr, (n, g) in zip(self.edit_lrs, mean_grads.items())} # mean_grads lr学习率是六个不同的mlp，自动更新

        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            edited_model = make_functional(edited_model, in_place=True)

        new_params = []
        for n, p in edited_model.named_parameters():
            if n in pset:
                new_params.append(p + updates[n])
            else:
                new_params.append(p)

        edited_model.update_params(new_params) # 更新参数

        if detach_history:
            new_model = self.model_constructor()
            new_model.load_state_dict(edited_model.state_dict())
            edited_model = new_model

        return MEND(edited_model, self.config, self.data, self.model_constructor, self.mend, edit_lrs=self.edit_lrs), info_dict


if __name__ == '__main__':
    import types

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.model.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.edit_lr = 0.0001

    config.mend = types.SimpleNamespace()
    config.mend.n_hidden = 1
    config.mend = config.mend.__dict__

    mend = MEND(model, config, lambda: copy.deepcopy(model)).cuda()
    import pdb; pdb.set_trace()
    mend.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000
    orig_logits = mend(x)
    edited = mend.edit(x, masks=torch.ones_like(x), labels=x)
    post_logits = mend(x)

    assert torch.allclose(orig_logits, post_logits)

    orig_param = [p for (n, p) in mend.model.named_parameters() if n == config.model.inner_params[-1]][0]
    edited_param = [p for (n, p) in edited.model.named_parameters() if n == config.model.inner_params[-1]][0]

    LOG.info((orig_param - edited_param).abs().max())
    edited.eval()
    LOG.info(mend(x, labels=x).loss, edited(x, labels=x).loss, edited.edit_loss_fn(edited(x).logits, x)["nll"])
    edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
    LOG.info(mend(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss)
