import faiss
import os
import pickle
import json
import random
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup
from models.trie import get_trie

def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0  # 只优化id为1～8的token
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

        
def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.tokenizer = tokenizer
    
        self._init_label_word()
        
    
    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        
        num_labels = len(label_word_idx)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]
            
            # for abaltion study
            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]
            
            if self.args.init_type_words:
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        
        self.word2label = continous_label_word # a continous list
            
                
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, so = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits
        output_embedding = result.hidden_states[-1]
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss
    
    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss


    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        
    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
        
    def ke_loss(self, logits, labels, so, input_ids):
        subject_embedding = []
        object_embedding = []
        neg_subject_embedding = []
        neg_object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))

            # random select the neg samples
            st_sub = random.randint(1, logits[i].shape[0] - 6)
            span_sub = random.randint(1, 5)
            st_obj = random.randint(1, logits[i].shape[0] - 6)
            span_obj = random.randint(1, 5)
            neg_subject_embedding.append(torch.mean(logits[i, st_sub:st_sub+span_sub], dim=0))
            neg_object_embedding.append(torch.mean(logits[i, st_obj:st_obj+span_obj], dim=0))
            
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        neg_subject_embedding = torch.stack(neg_subject_embedding)
        neg_object_embedding = torch.stack(neg_object_embedding)
        # trick , the relation ids is concated, 


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = logits[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        
        d_1 = torch.norm(subject_embedding + mask_relation_embedding - object_embedding, p=2) / bsz
        d_2 = torch.norm(neg_subject_embedding + real_relation_embedding - neg_object_embedding, p=2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1.*f(self.args.t_gamma - d_1) - f(d_2 - self.args.t_gamma)
        
        return loss

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }


class DialogueLitModel(BertLitModel):

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels) 
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)

    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
      

class GetEntityEmbeddingLitModel(BertLitModel):
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args, tokenizer)
        self.faissid2entityid = {}
        d, measure = self.model.config.hidden_size, faiss.METRIC_L2   
        self.index = faiss.IndexFlatL2(d)   # build the index
        self.cnt_batch = 0
        self.total_embedding = []


    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, _ = batch
        # last layer 
        hidden_states = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        entity_embedding = hidden_states[torch.arange(bsz), mask_idx].cpu()
        # use normalize or not ?
        # entity_embedding = F.normalize(entity_embedding, dim=-1, p = 2)
        self.total_embedding.append(entity_embedding)
        
        for i, l in zip(range(bsz), labels):
            self.faissid2entityid[i+self.cnt_batch] = l.cpu()
        self.cnt_batch += bsz


    def test_epoch_end(self, outputs) -> None:
        self.total_embedding = np.concatenate(self.total_embedding, axis=0)
        # self.index.train(self.total_embedding)
        print(faiss.MatrixStats(self.total_embedding).comments)
        self.index.add(self.total_embedding)
        faiss.write_index(self.index, os.path.join(self.args.data_dir, "faiss_dump.index"))
        with open(os.path.join(self.args.data_dir, "faissid2entityid.pkl") ,'wb') as file:
            pickle.dump(self.faissid2entityid, file)
        print(f"number of  entity embedding : {len(self.faissid2entityid)}")


class CombineEntityEmbeddingLitModel(BertLitModel):
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args, tokenizer)
        self.faissid2entityid = pickle.load(open(os.path.join(self.args.data_dir, "faissid2entityid.pkl") ,'rb'))
        self.index = faiss.read_index(os.path.join(self.args.data_dir, "faiss_dump.index"))
        self.dis2logits = distance2logits_2

    def _eval(self, batch):
        input_ids, attention_mask, labels, _ = batch

        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        hidden_states = result.hidden_states[-1]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        mask_embedding = np.array(hidden_states[torch.arange(bsz), mask_idx].cpu(), dtype=np.float32)
        topk = self.args.knn_topk
        D, I = self.index.search(mask_embedding, topk)
        D = torch.from_numpy(D).to(self.device)

        mask_logits = result.logits[:, :, self.word2label]
        mask_logits = mask_logits[torch.arange(bsz), mask_idx]
        
        label_size = len(self.word2label)
        entity_logits = torch.full((bsz, label_size), 1000.).to(self.device)
        
        for i in range(bsz):
            for j in range(topk):
                # filter entity in labels
                # if labels[i][self.faissid2entityid[I[i][j]]]: continue
                if entity_logits[i][self.faissid2entityid[I[i][j]]] == 1000.:
                    entity_logits[i][self.faissid2entityid[I[i][j]]] = D[i][j]
                # else:
                #     entity_logits[i][self.faissid2entityid[I[i][j]]] += D[i][j]
        entity_logits = self.dis2logits(entity_logits)
        mask_logits = torch.softmax(mask_logits, dim=-1)
        # get the entity ranks
        # filter the entity
        assert entity_logits.shape == (bsz, label_size)
        assert mask_logits.shape == (bsz, label_size)
        logits = combine_knn_and_vocab_probs(entity_logits, mask_logits, self.args.knn_lambda)
        return logits

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        logits = self._eval(batch)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff=0.5):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob


def distance2logits(D):
    return torch.softmax( -1. * torch.tensor(D) / 30., dim=-1)


def distance2logits_2(D, n=10):
    if torch.sum(D) != 0.0:
        distances = torch.exp(-D/n) / torch.sum(torch.exp(-D/n), dim=-1, keepdim=True)
    return distances


def distance2logits_dialogue(D, n=10):
    return torch.sigmoid( -1. * torch.tensor(D) / n)


class GetEntityEmbeddingLitModelDialogue(DialogueLitModel):
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args, tokenizer)
        self.faissid2entityid = {}
        d, measure = self.model.config.hidden_size, faiss.METRIC_L2   
        self.index = faiss.IndexFlatL2(d)   # build the index
        self.cnt_batch = 0
        self.total_embedding = []


    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids , labels = batch
        # last layer 
        hidden_states = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        entity_embedding = hidden_states[torch.arange(bsz), mask_idx].cpu()
        # use normalize or not ?
        # entity_embedding = F.normalize(entity_embedding, dim=-1, p = 2)
        self.total_embedding.append(entity_embedding)
        # self.index.add(np.array(entity_embedding, dtype=np.float32))
        
        for i, l in zip(range(bsz), labels):
            if len(l) > 0:
                self.faissid2entityid[i+self.cnt_batch] = (l.cpu() == 1).nonzero(as_tuple=False).view(-1)
            else:
                self.faissid2entityid[i+self.cnt_batch] = l.cpu()
        self.cnt_batch += bsz


    def test_epoch_end(self, outputs) -> None:
        self.total_embedding = np.concatenate(self.total_embedding, axis=0)
        # self.index.train(self.total_embedding)
        print(faiss.MatrixStats(self.total_embedding).comments)
        self.index.add(self.total_embedding)
        faiss.write_index(self.index, os.path.join(self.args.data_dir, "faiss_dump.index"))
        with open(os.path.join(self.args.data_dir, "faissid2entityid.pkl") ,'wb') as file:
            pickle.dump(self.faissid2entityid, file)
        print(f"number of  entity embedding : {len(self.faissid2entityid)}")


class CombineEntityEmbeddingLitModelDialogue(DialogueLitModel):
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args, tokenizer)
        self.faissid2entityid = pickle.load(open(os.path.join(self.args.data_dir, "faissid2entityid.pkl") ,'rb'))
        self.index = faiss.read_index(os.path.join(self.args.data_dir, "faiss_dump.index"))
        self.dis2logits = distance2logits_dialogue

    def _eval(self, batch):
        input_ids, attention_mask, token_type_ids , labels = batch

        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        hidden_states = result.hidden_states[-1]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        mask_embedding = np.array(hidden_states[torch.arange(bsz), mask_idx].cpu(), dtype=np.float32)
        topk = self.args.knn_topk
        D, I = self.index.search(mask_embedding, topk)
        D = torch.from_numpy(D).to(self.device)

        mask_logits = result.logits[:, :, self.word2label]
        mask_logits = mask_logits[torch.arange(bsz), mask_idx]
        
        label_size = len(self.word2label)
        entity_logits = torch.full((bsz, label_size), 1000.).to(self.device)
        
        for i in range(bsz):
            for j in range(topk):
                # filter entity in labels
                # if labels[i][self.faissid2entityid[I[i][j]]]: continue
                for idx in self.faissid2entityid[I[i][j]]:
                    if entity_logits[i][idx] == 1000.:
                        entity_logits[i][idx] = D[i][j]
                # else:
                #     entity_logits[i][self.faissid2entityid[I[i][j]]] += D[i][j]

        entity_logits = self.dis2logits(entity_logits)
        mask_logits = torch.sigmoid(mask_logits)
        # get the entity ranks
        # filter the entity
        assert entity_logits.shape == (bsz, label_size)
        assert mask_logits.shape == (bsz, label_size)
        logits = combine_knn_and_vocab_probs(entity_logits, mask_logits, self.args.knn_lambda)
        return logits

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self._eval(batch)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
