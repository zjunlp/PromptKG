import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from .retrocoop import combine_knn_and_vocab_probs

import faiss

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):

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

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        print("Constructing knn datastore by few-shot visual features and labels.")
        self.datastore, self.maskid2labelid = self.build_knn_datastore(cfg, clip_model.visual)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        clip_logits = logit_scale * image_features @ self.text_features.t()

        image_embeddings = np.array(image_features.cpu().detach(), dtype=np.float32)
        D, I = self.datastore.search(image_embeddings, self.cfg.RETRIEVE.topk)
        D = torch.from_numpy(D).to(image_features.device)

        knn_logits = torch.full((clip_logits.shape[0], clip_logits.shape[1]), 0.).to(image_features.device)

        for i in range(clip_logits.shape[0]):
            soft_knn_i = torch.softmax(D[i], dim=-1)
            for j in range(self.cfg.RETRIEVE.topk):
                knn_logits[i][self.maskid2labelid[I[i][j]]] += soft_knn_i[j]

        return combine_knn_and_vocab_probs(clip_logits, knn_logits, self.cfg.RETRIEVE.knn_lambda)


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
