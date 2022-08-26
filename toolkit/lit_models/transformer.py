from logging import debug
import math
import os
import copy
import random
import pytorch_lightning as pl
from tqdm import tqdm
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from transformers.utils.dummy_pt_objects import PrefixConstrainedLogitsProcessor

from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from functools import partial
from .utils import LabelSmoothSoftmaxCEV1, SparseMax, SparseMax_good

from models.trie import get_end_to_end_prefix_allowed_tokens_fn_hf

from models import Trie
from models.model import BartKGC
from models.trie import get_trie
from models import *

from pytorch_lightning.utilities.rank_zero import rank_zero_only

from typing import Callable, Iterable, List


__all__ = (
    "SimKGCLitModel",
    "KNNKGELitModel",
    "KNNKGEPretrainLitModel",
    "KGT5LitModel",
    "KGBartLitModel"
)

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[torch.tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List,
                    k=3, batch_size=256) :
    """
    save the hr_tensor and entity_tensor, evaluate in the KG.
    
    """
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        # rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, 

class SimKGCLitModel(BaseLitModel):
    def __init__(self, args, tokenizer=None, **kwargs):
        super().__init__( args)
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
        self.args = args
        self.model = SimKGCModel(args)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        outputs = self.model.compute_logits(output_dict=outputs, batch_dict=batch)
        # outputs = ModelOutput(**outputs)
        logits, labels = outputs.logits, outputs.labels
        bsz = len(batch['hr_token_ids'])
        assert logits.size(0) == bsz
        # head + relation -> tail
        loss = self.criterion(logits, labels)
        # tail -> head + relation
        loss += self.criterion(logits[:, :bsz].t(), labels)
        acc1, acc3 = accuracy(logits, labels, topk=(1,3))

        # self.logger.log_metrics(dict(acc1=acc1, acc3=acc3))
        return loss
    

    def _eval(self, batch, batch_idx):
        batch_size = len(batch['batch_data'])

        outputs = self.model(**batch)
        outputs = self.model.compute_logits(output_dict=outputs, batch_dict=batch)
        logits, labels = outputs.logits, outputs.labels
        loss = self.criterion(logits, labels)
        # losses.update(loss.item(), batch_size)

        acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
        return dict(loss=loss.detach().cpu(), acc1=acc1.detach().cpu(), acc3=acc3.detach().cpu())
    
    def validation_step(self, batch, batch_idx):
        return self._eval(batch, batch_idx)
    
    def init_entity_embedding(self):
        """
        get entity embedding in the KGs through description.
        
        """
        entity_dataloader = self.trainer.datamodule.get_entity_dataloader()
        entity_embedding = []

        for batch in tqdm(entity_dataloader, total=len(entity_dataloader), desc="Get entity embedding..."):
            for k ,v in batch.items():
                batch[k] = batch[k].to(self.device)
            entity_embedding += self.model.predict_ent_embedding(**batch).tolist()
        

        entity_embedding = torch.tensor(entity_embedding, device=self.device)
        self.entity_embedding = entity_embedding
    
    def on_test_epoch_start(self):
        self.init_entity_embedding()
    

    def test_step(self, batch, batch_idx):
        hr_vector = self.model(**batch)['hr_vector']
        scores = torch.mm(hr_vector, self.entity_embedding.t())
        bsz = len(batch['batch_data'])
        label = []
        for i in range(bsz):
            d = batch['batch_data'][i]
            inverse = d.inverse
            hr = tuple(d.hr)
            t = d.t
            label.append(t)
            idx = []
            if inverse:
                for hh in self.trainer.datamodule.filter_tr_to_h.get(hr, []):
                    if hh == t: continue
                    idx.append(hh)
            else:
                for hh in self.trainer.datamodule.filter_tr_to_h.get(hr, []):
                    if hh == t: continue
                    idx.append(hh)

            scores[i][idx] = -100
            # scores[i].index_fill_(0, idx, -1)
        _, outputs = torch.sort(scores, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1

        return dict(ranks = ranks)
    
    def test_epoch_end(self, outputs) -> None:
        ranks =  torch.cat([_['ranks'] for _ in outputs], axis=0)
        results = {}
        for h in [1,3,10]:
            results.update({f"hits{h}" : (ranks<=h).float().mean()})
        
        # self.logger.log_metrics(results)
        self.log_dict(results)
    
    def validation_epoch_end(self, outputs) -> None:
        acc1 = torch.cat([_['acc1'] for _ in outputs], dim=0)
        acc3 = torch.cat([_['acc3'] for _ in outputs], dim=0)
        self.log_dict(dict(acc1=acc1.mean().item(), acc3=acc3.mean().item()))
    

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        
        return parser


class LOGModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return torch.log(x)

class KNNKGELitModel(BaseLitModel):
    def __init__(self, args, tokenizer, **kwargs):
        super().__init__(args)
        self.save_hyperparameters(args)
        if args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            # self.loss_fn = nn.CrossEntropyLoss()
            # self.last_layer = nn.Sequential(
            #     SparseMax_good(),
            #     LOGModel(),
            # )
            # t = nn.NLLLoss()
            # self.loss_fn = lambda x,y: t(self.last_layer(x), y)
            self.loss_fn = SparseMax(100)
        self.best_acc = 0
        self.tokenizer = tokenizer
        self.model = KNNKGEModel.from_pretrained(args.model_name_or_path)

        # self.__dict__.update(data_config)
        # resize the word embedding layer

        self.decode = partial(decode, tokenizer=self.tokenizer)
        self.model.resize_token_embeddings(len(self.tokenizer) + kwargs['num_entity'] + kwargs['num_relation'])


    def on_fit_start(self) -> None:

        self.entity_id_st = self.tokenizer.vocab_size
        self.entity_id_ed = self.tokenizer.vocab_size + self.trainer.datamodule.num_entity
        self.realtion_id_st = self.tokenizer.vocab_size + self.trainer.datamodule.num_entity
        self.realtion_id_ed = self.tokenizer.vocab_size + self.trainer.datamodule.num_entity + self.trainer.datamodule.num_relation

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # embed();exit()
        # print(self.optimizers().param_groups[1]['lr'])
        label = batch.pop("label")
        input_ids = batch['input_ids']
        logits = self.model(**batch, return_dict=True).logits

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_logits = logits[torch.arange(bs), mask_idx][:, self.entity_id_st:self.entity_id_ed]

        assert mask_idx.shape[0] == bs, "only one mask in sequence!"

        loss = self.loss_fn(mask_logits, label)

        # if batch_idx == 0:
        #     print('\n'.join(self.decode(batch['input_ids'][:4])))
        

        return loss

    def _eval(self, batch, batch_idx, ):
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')
        filter_entity_ids = batch.pop('filter_entity_ids', [[] for _ in range(input_ids.shape[0])])
        my_keys = list(batch.keys())
        for k in my_keys:
            if k not in ["input_ids", "attention_mask", "token_type_ids"]:
                batch.pop(k)
        logits = self.model(**batch, return_dict=True).logits[:, :, self.entity_id_st:self.entity_id_ed]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx]
        # get the entity ranks
        # filter the entity
        # assert filter_entity_ids[0][label[0]], "correct ids must in filiter!"
        # labels[torch.arange(bsz), label] = 0
        
        # assert logits.shape == labels.shape
        for i in range(logits.shape[0]):
            # if len(filter_entity_ids[i]) == 0: continue
            try:
                logits[i][filter_entity_ids[i]] = -100
            except:
                import IPython; IPython.embed(); exit(1)
        # logits += labels * -100 # mask entityj
        # for i in range(bsz):
        #     logits[i][labels]

        _, outputs = torch.sort(logits, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        

        return dict(ranks = np.array(ranks))

    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])
        total_ranks = ranks.shape[0]

        if not self.args.pretrain:
            l_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2)))]
            r_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2))) + 1]
            self.log("Eval/lhits10", (l_ranks<=10).mean())
            self.log("Eval/rhits10", (r_ranks<=10).mean())

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval/hits10", hits10)
        self.log("Eval/hits20", hits20)
        self.log("Eval/hits3", hits3)
        self.log("Eval/hits1", hits1)
        self.log("Eval/mean_rank", ranks.mean())
        self.log("Eval/mrr", (1. / ranks).mean())
        self.log("hits10", hits10, prog_bar=True)
        self.log("hits1", hits1, prog_bar=True)

            
    

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # ranks = self._eval(batch, batch_idx)
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))

        return result

    def test_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

       
        self.log("Test/hits10", hits10)
        self.log("Test/hits20", hits20)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mean_rank", ranks.mean())
        self.log("Test/mrr", (1. / ranks).mean())

    
    def _freaze_attention(self):
        for k, v in self.model.named_parameters():
            if "word" not in k:
                v.requires_grad = False
            else:
                print(k)
    
    def _freaze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser

class KNNKGEPretrainLitModel(KNNKGELitModel):
    def __init__(self, args, tokenizer, **kwargs):
        super().__init__(args, tokenizer, **kwargs)
        self._freaze_attention()
    
    def on_test_epoch_start(self) -> None:
        file_folder = f"output/{self.args.dataset}/knnkge_pretrain_model"
        print(f"saving the model to {file_folder}...")

        self.model.save_pretrained(file_folder)
        self.tokenizer.save_pretrained(file_folder)



class KGT5LitModel(BaseLitModel):
    def __init__(self, args, tokenizer=None, **kwargs):
        super().__init__(args)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=['model'])
        # bug, cannot fix
        self.args = args
        self.model = self._init_model()
        self.best_acc = 0
        
        self.tokenizer = tokenizer
        self.entity_trie = None
        if args.prefix_tree_decode:
            self.entity_trie = get_trie(args, tokenizer=tokenizer)
        
        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.decode = partial(decode, tokenizer=self.tokenizer)


        self._eval = self._eval_1
    
    def _init_model(self):
        model = T5KBQAModel.from_pretrained(self.args.model_name_or_path)
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # batch.pop("filter_ent_ids")
        bsz = batch['input_ids'].shape[0]
        loss = self.model(**batch).loss
        self.log("Train/loss", loss)
        return loss
        
    def _get_ranks(self, outputs, labels):
        bsz = len(labels)
        tmp = []
        ranks = []
        for i in range(bsz):
            # get the real entity
            label = labels[i]
            output = outputs[i]
            # filter entities to ignore
            has_flag = False
            for j in range(len(output)):
                for o, l in zip(label, output[j][1:]):
                    if o != l: break
                    if l == self.tokenizer.eos_token_id:
                        has_flag = True
                        ranks.append(j+1)
                        break
                if has_flag: break
                # if label in output[j]:
                #     has_flag = True
                #     ranks.append(real_idx)
                #     break
            #TODO more youmei solution
            if not has_flag: ranks.append(10000)
        return ranks
    
    def _eval_normal(self, batch, batch_idx, ):
        labels = batch.pop("labels")
        # decoder_input_ids = batch.pop("decoder_input_ids")
        # ent id for filter
        # ent = batch.pop("filter_ent_ids")
        bsz = batch['input_ids'].shape[0]

        topk = self.args.beam_size
        prefix_allowed_tokens_fn = None
        if self.entity_trie:
            prefix_allowed_tokens_fn = lambda batch_id, sent: self.entity_trie.get(sent.tolist())
        outputs = self.model.generate(
            **batch,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=topk, num_return_sequences=topk,
            top_p=1.,
            max_length=64,
            use_cache=True,
        ).view(bsz, topk, -1).cpu()
        src = self.decode(labels)
        outputs = [self.decode(o) for o in outputs]
        ranks = []
        for i in range(bsz):
            in_flag = False
            for j in range(topk):
                if outputs[i][j] == src[i]:
                    ranks.append(j)
                    in_flag = True
                    break
            if not in_flag: ranks.append(10000)

        return dict(ranks=ranks)
            



        # if batch_idx == 10:
    
        # ranks = self._get_ranks(outputs, labels, ent)
        # if batch_idx % 10 == 0:
        #     generated_text = [_[:5].cpu().tolist() for _ in outputs]
        #     outputs_ = self.model.generate(**batch, forced_bos_token_id=self.tokenizer.bos_token_id, use_cache=True, max_length=40).cpu().tolist()
        #     # # self.log("conditonal output",generated_text)
        #     input_text = self.decode(output_ids=batch['input_ids'])
        #     # label_text = self.decode(output_ids=labels)
        #     label_text = labels
        #     # outputs_ = 

        #     # log info
        #     columns = ["input", "label", "output", "prefix output"]
        #     data = [[input_text[i], label_text, outputs_[i], generated_text[i] ] for i in range(bsz)]
        #     self.logger.log_text(key="samples", columns=columns, data=data)
        #         # print(f"origin input : \t{input_text[i]}")
        #         # print(f"model output : \t{outputs_[i]}")
        #         # print(f"model constrined output : \t{generated_text[i]}")
        #         # # print(f"true label : \t{label_text[i]}")
        #         # print("*"*20)

        # return dict(outputs=outputs, labels=labels.cpu())


    def _eval_1(self, batch, batch_idx):
        ranks_in_batch = {
            "unfiltered": list(),
            "filtered": list(),
        }

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        input_ids_repeated = torch.repeat_interleave(
                input_ids, len(self.dataset.entity_strings), dim=0
            )
        attention_mask_repeated = torch.repeat_interleave(
            attention_mask, len(self.dataset.entity_strings), dim=0
        )
        tokenized_entities = self.trainer.datamodule.tokenized_entities.input_ids.to(
            self.device
        )
        all_entities_repeated = tokenized_entities.repeat([self.batch_size, 1])
        summed_logit_chunks = []
        # process chunk by chunk
        self.chunk_size = self.args.chunk_size
        for chunk_number in range(
            math.ceil(len(input_ids_repeated) / self.chunk_size)
        ):
            chunk_start = self.chunk_size * chunk_number
            chunk_end = min(
                self.chunk_size * (chunk_number + 1), len(input_ids_repeated)
            )
            current_chunk_size = chunk_end - chunk_start
            outputs_chunk = self.model(
                input_ids=input_ids_repeated[chunk_start:chunk_end],
                attention_mask=attention_mask_repeated[chunk_start:chunk_end],
                labels=all_entities_repeated[chunk_start:chunk_end],
            )
            logits_chunk = outputs_chunk.logits
            soft_logits_chunk = torch.log_softmax(logits_chunk, dim=2)
            coordinates = all_entities_repeated[chunk_start:chunk_end].view(current_chunk_size, -1, 1)
            # set padded logits to zero
            padded_mask = (coordinates == 0).squeeze()
            soft_logits_chunk[padded_mask] = 0
            needed_soft_logits_chunk = torch.gather(
                soft_logits_chunk,
                2,
                coordinates
            ).view(current_chunk_size, -1)

            summed_logits = torch.sum(needed_soft_logits_chunk, dim=1)
            summed_logit_chunks.append(summed_logits)
        summed_logits = torch.cat(summed_logit_chunks)

        for summed_logits_per_triple, hr_pair, label in zip(
            summed_logits.split(len(self.dataset.entity_strings)), batch['hr_pairs'].cpu().tolist(), batch['target_entity_id'].cpu().tolist()
        ):
            # todo: currently we are calculating best rank on equality
            #  change to mean
            arg_sorted = torch.argsort(summed_logits_per_triple, descending=True)
            entity_id = label
            rank = (
                (arg_sorted == entity_id)
                .nonzero(as_tuple=True)[0]
                .item()
            )
            print(rank)
            ranks_in_batch["unfiltered"].append(rank)

            # now filter
            true_score = summed_logits_per_triple[entity_id].clone()
            filter_entity_id = self.filter_hr_to_ent[hr_pair].remove(entity_id)
            summed_logits_per_triple[filter_entity_id] = -float("inf")

            summed_logits_per_triple[entity_id] = true_score
            arg_sorted = torch.argsort(summed_logits_per_triple, descending=True)
            rank = (
                (arg_sorted == entity_id)
                    .nonzero(as_tuple=True)[0]
                    .item()
            )
            ranks_in_batch["filtered"].append(rank)
    #     ranks["filtered"].extend(ranks_in_batch["filtered"])
    #     ranks["unfiltered"].extend(ranks_in_batch["unfiltered"])
    # for setting, list_of_ranks in ranks.items():
    #     ranks[setting] = np.array(list_of_ranks, dtype=np.float32) + 1
        return dict(unfiltered_ranks = np.array(ranks_in_batch['unfiltered']+1), filtered_ranks = np.array(ranks_in_batch['filtered'])+1)

    def validation_step(self, batch, batch_idx):
        result = self._eval_normal(batch, batch_idx)
        # self.log("Eval/loss", np.mean(ranks))
        return result

    def validation_epoch_end(self, outputs) -> None:
            
        # labels = [_ for o in outputs for _ in o['labels']]
        # outputs = [_ for o in outputs for _ in o['outputs']]
        keys = outputs[0].keys()
        ranks = np.concatenate([o['ranks'] for o in outputs], axis=0)
        # ranks = self._get_ranks(outputs, labels)
        # ranks = np.array(ranks)
        # hits20 = (ranks<=20).mean()
        # hits10 = (ranks<=10).mean()
        # hits3 = (ranks<=3).mean()
        # hits1 = (ranks<=1).mean()

        for hit in [1, 3, 10]:
            r = (ranks <= hit).mean()
            self.log(f"hits{hit}", r, prog_bar=True)

        # self.log("Eval/hits10", hits10, prog_bar=True, on_epoch=True)
        # self.log("Eval/hits1", hits1)
        # self.log("Eval/hits3", hits3)
        # self.log("Eval/hits20", hits20)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # ranks = self._eval(batch, batch_idx)
        result = self.validation_step(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))

        return result
        # return {"test_rank": np.array(ranks)}

    def test_epoch_end(self, outputs) -> None:
        self.validation_epoch_end(outputs)
        

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]


        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--output_full_sentence", type=int, default=0, help="decode full sentence not entity")
        parser.add_argument("--entity_token", type=str, default="<entity>", help="")
        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--prefix_tree_decode", type=int, default=1, help="")
        parser.add_argument("--beam_size", type=int, default=10, help="")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        
        return parser
    

class KGBartLitModel(KGT5LitModel):
    def __init__(self, args, tokenizer=None, **kwargs) -> None:
        super().__init__(args, tokenizer)

        if args.use_ce_loss:
            self.model.loss_fn = nn.CrossEntropyLoss()
        
    def _init_model(self):
        model = BartKGC.from_pretrained(self.args.model_name_or_path)
        if self.args.use_ce_loss:
            model.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=self.args.label_smoothing)
        else:
            model.loss_fn = SparseMax(20)
        return model
    
    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--prefix_tree_decode", type=int, default=0, help="")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        parser.add_argument("--beam_size", type=int, default=60, help="")
        parser.add_argument("--use_ce_loss", type=int, default=0, help="")
        
        return parser
    
    def _eval_normal(self, batch, batch_idx, ):
        #TODO add filiter
        labels = batch.pop("labels")
        batch_data = batch.pop("batch_data")
        # decoder_input_ids = batch.pop("decoder_input_ids")
        # ent id for filter
        # ent = batch.pop("filter_ent_ids")
        bsz = batch['input_ids'].shape[0]

        hr_t = self.trainer.datamodule.filter_hr_to_t
        tr_h = self.trainer.datamodule.filter_tr_to_h

        entity2id = self.trainer.datamodule.entity2id

        topk = self.args.beam_size
        prefix_allowed_tokens_fn = None
        if self.entity_trie:
            prefix_allowed_tokens_fn = lambda batch_id, sent: self.entity_trie.get(sent.tolist())
        outputs = self.model.generate(
            **batch,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=topk, num_return_sequences=topk,
            top_p=1.,
            max_length=64,
            use_cache=True,
        ).view(bsz, topk, -1).cpu()
        src = self.decode(labels)
        outputs = [self.decode(o) for o in outputs]
        ranks = []
        for i in range(bsz):
            in_flag = False
            cnt = 1
            for j in range(topk):
                if outputs[i][j] == src[i]:
                    ranks.append(cnt)
                    in_flag = True
                    break
                if outputs[i][j] in entity2id:
                    target_id = entity2id[outputs[i][j]]
                    if not batch_data[i].inverse:
                        if target_id in hr_t[batch_data[i].hr]:
                            continue
                    else:
                        if target_id in tr_h[batch_data[i].hr]:
                            continue
                
                cnt += 1

            if not in_flag: ranks.append(10000)

        return dict(ranks=ranks)
