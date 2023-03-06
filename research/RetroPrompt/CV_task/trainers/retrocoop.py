import os
import os.path as osp
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import faiss

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def combine_knn_and_vocab_probs(clip_logits, knn_logits, coeff=0.5):
    mask_logits = torch.softmax(clip_logits, dim=-1)
    combine_probs = torch.stack([mask_logits, knn_logits], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob


class knnLoss(nn.Module):
    def __init__(self):
        super(knnLoss, self).__init__()

    def loss(self, logits, knn_logits, targets, coeff):
        loss = F.cross_entropy(logits, targets, reduction="mean")

        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        knn_loss = F.nll_loss(torch.clamp(torch.log(p), min=-100),
            targets, reduction="mean")

        loss = loss + torch.mul(loss, knn_loss * coeff)
        
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, knn_logits,
        targets, coeff
    ):
        loss = self.loss(
            pred_logits, knn_logits, targets,
            coeff)
        return loss


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.RETROCOOP.N_CTX
        ctx_init = cfg.TRAINER.RETROCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.RETROCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.RETROCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, datastore, maskid2labelid):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.datastore = datastore
        self.maskid2labelid = maskid2labelid
        self.topk = cfg.RETRIEVE.topk

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        clip_logits = logit_scale * image_features @ text_features.t()

        image_embeddings = np.array(image_features.cpu().detach(), dtype=np.float32)
        D, I = self.datastore.search(image_embeddings, self.topk)
        D = torch.from_numpy(D).to(image_features.device)

        knn_logits = torch.full((clip_logits.shape[0], clip_logits.shape[1]), 0.).to(image_features.device)

        soft_knn = torch.full((clip_logits.shape[0], self.topk), 0.).to(image_features.device)

        labelid = np.zeros( (50, 16) )

        batch_idx = 0

        import ipdb; ipdb.set_trace()

        for i in range(clip_logits.shape[0]):
            soft_knn[i] = torch.softmax(D[i], dim=-1)
            for j in range(self.topk):
                knn_logits[i][self.maskid2labelid[I[i][j]]] += soft_knn[i][j]
                labelid[i][j] = self.maskid2labelid[I[i][j]]
            for k in range(10):
                if labelid[i][k] == batch_idx:
                    print(i)

        combine_logits = combine_knn_and_vocab_probs(clip_logits, knn_logits, 0.9)
        import ipdb; ipdb.set_trace()
        return clip_logits, knn_logits


@TRAINER_REGISTRY.register()
class RetroCoOp(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.RETROCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_knn_datastore(self, cfg, image_encoder):

        if cfg.RETRIEVE.load_cache == False or cfg.RETRIEVE.update_cache == True:
            os.makedirs(cfg.CACHE_DIR, exist_ok=True)

            cache_keys = []
            cache_values = []
            image_encoder.to(self.device)

            with torch.no_grad():
                # Data augmentation for the cache model
                for augment_idx in range(cfg.RETRIEVE.augment_epoch):
                    train_features = []

                    print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg.RETRIEVE.augment_epoch))
                    for i, batch in enumerate(tqdm(self.train_loader_x)):
                        images = batch["img"]
                        target = batch["label"]
                        images = images.to(self.device)
                        image_features = image_encoder(images)
                        train_features.append(image_features)
                        if augment_idx == 0:
                            target = target.to(self.device)
                            cache_values.append(target)
                    cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)
            cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

            torch.save(cache_keys, cfg.CACHE_DIR + '/keys_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")
            torch.save(cache_values, cfg.CACHE_DIR + '/values_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")

        else:
            cache_keys = torch.load(cfg.CACHE_DIR + '/keys_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")
            cache_values = torch.load(cfg.CACHE_DIR + '/values_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")

        cache_keys = cache_keys.permute(1, 0)
        n, d = cache_keys.shape[0], cache_keys.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(cache_keys.cpu().numpy())
        maskid2labelid = torch.argmax(cache_values, dim=-1)

        return index, maskid2labelid

    def after_epoch(self):
        super().after_epoch()

        if self.cfg.RETRIEVE.update_cache == True:
            if (self.epoch + 1)  % self.cfg.RETRIEVE.update_epoch == 0:
                print("Updating knn datastore by few-shot visual features and labels.")
                self.model.datastore, self.model.maskid2labelid = self.build_knn_datastore(self.cfg, self.model.image_encoder)
            
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.RETROCOOP.PREC == "fp32" or cfg.TRAINER.RETROCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Constructing knn datastore by few-shot visual features and labels.")
        datastore, maskid2labelid = self.build_knn_datastore(cfg, clip_model.visual)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, datastore, maskid2labelid)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.RETROCOOP.PREC == "amp" else None

        self.loss_func = knnLoss()

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.RETROCOOP.PREC
        if prec == "amp":
            with autocast():
                clip_logits, knn_logits = self.model(image)
                if self.cfg.RETRIEVE.train_with_knn:
                    loss = self.loss_func(clip_logits, knn_logits, label, self.cfg.RETRIEVE.beta)
                else:
                    loss = F.cross_entropy(clip_logits, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            clip_logits, knn_logits = self.model(image)
            if self.cfg.RETRIEVE.train_with_knn and knn_logits is not None: 
                loss = self.loss_func(clip_logits, knn_logits, label, self.cfg.RETRIEVE.beta)
            else:
                loss = F.cross_entropy(clip_logits, label)
            self.model_backward_and_update(loss)

        output = combine_knn_and_vocab_probs(clip_logits, knn_logits, self.cfg.RETRIEVE.knn_lambda)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        clip_logits, knn_logits = self.model(input)
        return combine_knn_and_vocab_probs(clip_logits, knn_logits, self.cfg.RETRIEVE.knn_lambda)

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
