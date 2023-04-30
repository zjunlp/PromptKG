import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import EditBatchSampler, dict_to
import random


POSITIVE_CLASS = "SUPPORTS"


class BinaryAugmentedKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        max_length=32
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.config = config

        def extract(d):
            extracted = {k: d[k] for k in ["logit", "input", "prediction", "alternatives", "filtered_rephrases"]}
            extracted["label"] = d["output"][0]["answer"] # 原来模型的回答为label
            return extracted

        with jsonlines.open(data_path) as f:
            for d in f:
                if len(d["alternatives"]) > 0 and len(d["filtered_rephrases"]) > 0:
                    self.data.append(extract(d))

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obj = self.data[item]
        rephrase = random.choice(self.data[item]["filtered_rephrases"])
        output = {
            "label": obj["label"] == POSITIVE_CLASS, # 模型output，这个label和pred存在疑问
            "src": obj["input"], # 原来的句子
            "rephrase": rephrase, # 改写的句子
            "pred": obj["prediction"] == POSITIVE_CLASS, # 预测结果
            "alt": obj["alternatives"][0] == POSITIVE_CLASS, # 替换
            "cond_flip": "{} >> {} || {}".format(
                obj["prediction"],
                obj["alternatives"][0],
                obj["input"],
            ),
            "cond_orig": "{} >> {} || {}".format(
                obj["prediction"],
                obj["prediction"],
                obj["input"],
            ),
            "logit": obj["logit"],
        }

        return output

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        rephrase = [batch[-1]["rephrase"]]

        flip_label = np.random.uniform() > 0.5 # 随机采样
        predictions = [b["pred"] for b in batch]
        labels = [b["label"] for b in batch]
        labels[-1] = predictions[-1]  # the last element in the batch is special (the edit element)
        cond = [batch[-1]["cond_orig"]]
        if flip_label: # 一般的概率替换
            labels[-1] = batch[-1]["alt"]
            cond = [batch[-1]["cond_flip"]]

        batches = {}
        for k1, v1 in {"": src, "cond_": cond, "rephrase_": rephrase}.items():
            encoded = self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
            for k2, v2 in encoded.items():
                batches[f"{k1}{k2}"] = v2

        batches["predictions"] = torch.tensor(predictions).long().view(-1, 1)
        batches["labels"] = torch.tensor(labels).long().view(-1, 1)
        batches["raw"] = batch
        return batches

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(n, memorize_mode=self.config.single_batch, seed=self.config.seed)
        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            assert len(edit_idxs) == 1
            idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in idxs])

            pass_keys = ["input_ids", "attention_mask", "labels"]
            edit_inner = {k: v[-1:] for k, v in toks.items() if k in pass_keys} # src的tokenizer，取最后一个edit_idx的
            if self.config.data.rephrase:
                edit_outer = {}
                edit_outer["input_ids"] = toks["rephrase_input_ids"]
                edit_outer["attention_mask"] = toks["rephrase_attention_mask"]
                edit_outer["labels"] = edit_inner["labels"] # 重写的edit的tokenizer
            else:
                edit_outer = edit_inner
            loc = {k: v[:-1] for k, v in toks.items() if k in pass_keys} # loc的tokenizer
            cond = {"input_ids": toks["cond_input_ids"], "attention_mask": toks["cond_attention_mask"]}

            batch = {
                "edit_inner": edit_inner, # src
                "edit_outer": edit_outer, # rephrase
                "loc": loc, # 衡量无关edit的变化程度
                "cond": cond # cond的tokenizer，cond有一半概率用ori一般概率用替换的
            }
            yield dict_to(batch, self.config.device)
