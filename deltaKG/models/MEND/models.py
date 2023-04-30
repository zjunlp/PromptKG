import transformers
import torch
import copy
import json
import torch.nn as nn
import re
import os

import logging
from utils import scr
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel, BertForSequenceClassification

LOG = logging.getLogger(__name__)


class CastModule(nn.Module):
    def __init__(self, module: nn.Module, in_cast: torch.dtype = torch.float32, out_cast: torch.dtype = None):
        super().__init__()

        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        else:
            return obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = self.cast(outputs, self.out_cast)
        elif isinstance(outputs, tuple):
            outputs = tuple(self.cast(o, self.out_cast) for o in outputs)
        else:
            raise RuntimeError(f"Not sure how to cast type {type(outputs)}")
        return outputs

    def extra_repr(self):
        return f"in_cast: {self.in_cast}\nout_cast: {self.out_cast}"


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained(model_name, cache_dir=scr())
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        return self.classifier(self.model(*args, **filtered_kwargs)[1])

def read_config(file):
    with open(file, 'r') as f:
        return json.load(f)

def get_model(config):
    if config.model.class_name == "BertClassifier":
        model = BertClassifier(config.model.name)
    elif config.model.class_name == "BertMaskedLM":
        model = BertForMaskedLM.from_pretrained(config.model.from_pretrain if config.model.from_pretrain is not None else config.model.name)
    elif config.model.class_name == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained(config.model.from_pretrain if config.model.from_pretrain is not None else config.model.name)
    elif config.model.class_name == "LMKE":
        lm_model = BertModel.from_pretrained('bert-base-uncased')
        model = LMKE(lm_model, **read_config(os.path.join(config.model.from_pretrain, "config.json")))
        model.load_state_dict(torch.load(os.path.join(config.model.from_pretrain, "checkpoint.pt")))
    else:
        ModelClass = getattr(transformers, config.model.class_name)
        LOG.info(f"Loading model class {ModelClass} with name {config.model.name} from cache dir {scr()}")
        model = ModelClass.from_pretrained(config.model.name, cache_dir=scr())

    if config.model.pt is not None:
        LOG.info(f"Loading model initialization from {config.model.pt}")
        state_dict = torch.load(config.model.pt, map_location="cpu")

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            LOG.info("Default load failed; stripping prefix and trying again.")
            state_dict = {re.sub("^model.", "", k): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)

        LOG.info("Loaded model initialization")

    if config.dropout is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = config.dropout
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = config.dropout
                    n_reset += 1

            if hasattr(m, "activation_dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = config.dropout
                    n_reset += 1

        LOG.info(f"Set {n_reset} dropout modules to p={config.dropout}")

    param_names = [n for n, _ in model.named_parameters()]
    bad_inner_params = [p for p in config.model.inner_params if p not in param_names]
    if len(bad_inner_params) != 0:
        raise ValueError(f"Params {bad_inner_params} do not exist in model of type {type(model)}.")

    if config.no_grad_layers is not None:
        if config.half:
            model.bfloat16()

        def upcast(mod):
            modlist = None
            for child in mod.children():
                if isinstance(child, nn.ModuleList):
                    assert modlist is None, f"Found multiple modlists for {mod}"
                    modlist = child
            if modlist is None:
                raise RuntimeError("Couldn't find a ModuleList child")

            LOG.info(f"Setting {len(modlist) - config.no_grad_layers} modules to full precision, with autocasting")
            modlist[config.no_grad_layers:].to(torch.float32)
            modlist[config.no_grad_layers] = CastModule(modlist[config.no_grad_layers])
            modlist[-1] = CastModule(modlist[-1], in_cast=torch.float32, out_cast=torch.bfloat16)

        parents = []
        if hasattr(model, "transformer"):
            parents.append(model.transformer)
        if hasattr(model, "encoder"):
            parents.append(model.encoder)
        if hasattr(model, "decoder"):
            parents.append(model.decoder)
        if hasattr(model, "model"):
            parents.extend([model.model.encoder, model.model.decoder])

        for t in parents:
            t.no_grad_layers = config.no_grad_layers
            if config.half:
                upcast(t)

        if config.half:
            idxs = []
            for p in config.model.inner_params:
                for comp in p.split('.'):
                    if comp.isdigit():
                        idxs.append(int(comp))
            max_idx, min_idx = str(max(idxs)), str(config.no_grad_layers)
            for pidx, p in enumerate(config.model.inner_params):
                comps = p.split('.')
                if max_idx in comps or min_idx in comps:
                    index = comps.index(max_idx) if max_idx in comps else comps.index(min_idx)
                    comps.insert(index + 1, 'underlying')
                    new_p = '.'.join(comps)
                    LOG.info(f"Replacing config.model.inner_params[{pidx}] '{p}' -> '{new_p}'")
                    config.model.inner_params[pidx] = new_p

    return model


def get_tokenizer(config):
    tok_name = config.model.tokenizer_name if config.model.tokenizer_name is not None else config.model.name
    return getattr(transformers, config.model.tokenizer_class).from_pretrained(tok_name)


num_deg_features = 2
class LMKE(nn.Module):
	def __init__(self, lm_model, n_ent, n_rel, add_tokens, contrastive):
		super().__init__()

		self.lm_model_given = lm_model
		self.lm_model_target = copy.deepcopy(lm_model)
		self.lm_model_classification = copy.deepcopy(lm_model)

		self.n_ent = n_ent
		self.n_rel = n_rel
		self.hidden_size = lm_model.config.hidden_size

		self.add_tokens = add_tokens
		self.contrastive = contrastive 

		self.ent_embeddings = torch.nn.Embedding(n_ent, self.hidden_size)
		self.rel_embeddings = torch.nn.Embedding(n_rel, self.hidden_size)

		self.ent_embeddings_transe = torch.nn.Embedding(n_ent, self.hidden_size)
		self.rel_embeddings_transe = torch.nn.Embedding(n_rel, self.hidden_size)

		self.mask_embeddings = torch.nn.Embedding(3, self.hidden_size)

		self.classifier = torch.nn.Linear(self.hidden_size, 2)

		self.confidence_gate = torch.nn.Linear(self.hidden_size, 1)
	
		self.rel_classifier = torch.nn.Linear(self.hidden_size, n_rel)
		self.ent_classifier = torch.nn.Linear(self.hidden_size, n_ent)
	

		self.sim_classifier = nn.Sequential(nn.Linear(self.hidden_size * 4 + num_deg_features, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, 1))

		self.ensemble_weights_pred_h = nn.Linear(num_deg_features, 2)
		self.ensemble_weights_pred_r = nn.Linear(num_deg_features, 2)
		self.ensemble_weights_pred_t = nn.Linear(num_deg_features, 2)

	def forward(self, inputs, positions, mode, triple_degrees=None):

		batch_size = len(positions)

		if mode == 'triple_classification':
			lm_model = self.lm_model_classification
		else:
			lm_model = self.lm_model_given

		device = lm_model.device

		h_idx = torch.LongTensor([positions[i]['head'][0] for i in range(batch_size)]).to(device)
		h_pos = torch.LongTensor([positions[i]['head'][1] for i in range(batch_size)]).to(device)
		r_idx = torch.LongTensor([positions[i]['rel'][0]  for i in range(batch_size)]).to(device)
		r_pos = torch.LongTensor([positions[i]['rel'][1]  for i in range(batch_size)]).to(device)
		t_idx = torch.LongTensor([positions[i]['tail'][0] for i in range(batch_size)]).to(device)
		t_pos = torch.LongTensor([positions[i]['tail'][1] for i in range(batch_size)]).to(device)

		if not self.add_tokens:
			input_ids = inputs.pop('input_ids')
			input_embeds = self.lm_model_given.embeddings.word_embeddings(input_ids).squeeze(1)
			
			if self.contrastive:
				if mode == 'link_prediction_h':
					mask_emb = self.mask_embeddings(torch.LongTensor([0]).cuda())
				elif mode == 'link_prediction_r':
					mask_emb = self.mask_embeddings(torch.LongTensor([1]).cuda())
				elif mode == 'link_prediction_t':
					mask_emb = self.mask_embeddings(torch.LongTensor([2]).cuda())

			for i in range(batch_size):
				if mode != 'link_prediction_h':
					input_embeds[i, h_pos[i], :] = self.ent_embeddings(h_idx[i])
				else:
					if self.contrastive:
						input_embeds[i, h_pos[i], :] = mask_emb
				if mode != 'link_prediction_r':
					input_embeds[i, r_pos[i], :] = self.rel_embeddings(r_idx[i])
				else:
					if self.contrastive:
						input_embeds[i, r_pos[i], :] = mask_emb
				if mode != 'link_prediction_t':
					input_embeds[i, t_pos[i], :] = self.ent_embeddings(t_idx[i])
				else:
					if self.contrastive:
						input_embeds[i, t_pos[i], :] = mask_emb

			inputs['inputs_embeds'] = input_embeds

		logits = lm_model(**inputs) 
		

		h_emb_list = []
		r_emb_list = []
		t_emb_list = []

		try:
			triple_embs = logits[1]
		except:
			triple_embs = logits[0][:, 0, :]

		for i in range(batch_size):
			h_emb_list.append(logits[0][i, h_pos[i], :].unsqueeze(0))
			r_emb_list.append(logits[0][i, r_pos[i], :].unsqueeze(0))
			t_emb_list.append(logits[0][i, t_pos[i], :].unsqueeze(0))

		h_embs = torch.cat(h_emb_list, dim=0)
		r_embs = torch.cat(r_emb_list, dim=0)
		t_embs = torch.cat(t_emb_list, dim=0)


		# Triple classification 
		if mode == 'triple_classification':
			preds = self.classifier(torch.cat([triple_embs], dim=-1))
			return preds 

		if self.contrastive:
			# return logits of masked positions
			if mode == 'link_prediction_h':
				return h_embs
			elif mode == 'link_prediction_r':
				return r_embs
			elif mode == 'link_prediction_t':
				return t_embs
		else:
			# MEM-KGE, a masked variant of LMKE 
			if mode == 'link_prediction_h':
				preds = self.ent_classifier(h_embs)
			elif mode == 'link_prediction_r':
				preds = self.rel_classifier(r_embs)
			elif mode == 'link_prediction_t':
				preds = self.ent_classifier(t_embs)

			return preds


	def forward_without_text(self, inputs, positions):
		batch_size = len(positions)
		device = self.lm_model_given.device

		h_idx = torch.LongTensor([positions[i]['head'][0] for i in range(batch_size)]).to(device)
		r_idx = torch.LongTensor([positions[i]['rel'][0]  for i in range(batch_size)]).to(device)
		t_idx = torch.LongTensor([positions[i]['tail'][0] for i in range(batch_size)]).to(device)

		h_emb_list = []
		r_emb_list = []
		t_emb_list = []
		for i in range(batch_size):
			h_emb_list.append(self.ent_embeddings_transe(h_idx[i]).unsqueeze(0))
			r_emb_list.append(self.rel_embeddings_transe(r_idx[i]).unsqueeze(0))
			t_emb_list.append(self.ent_embeddings_transe(t_idx[i]).unsqueeze(0))

		
		h_embs = torch.cat(h_emb_list, dim=0)
		r_embs = torch.cat(r_emb_list, dim=0)
		t_embs = torch.cat(t_emb_list, dim=0)

		return h_embs, r_embs, t_embs


	def score_triples_transe(self, h_embs, r_embs, t_embs):
		scores = (h_embs + r_embs - t_embs).square().sum(dim=-1).sqrt()
		
		return scores

	def score_triples_rotate(self, h_embs, r_embs, t_embs, mode):
		h_embs_re = h_embs[:, :, :, 0]
		h_embs_im = h_embs[:, :, :, 1]
		r_embs_re = r_embs[:, :, :, 0]
		r_embs_im = r_embs[:, :, :, 1]
		t_embs_re = t_embs[:, :, :, 0]
		t_embs_im = t_embs[:, :, :, 1]

		if mode in ['link_prediction_t', 'tail']:
			h_multiply_r_re = h_embs_re * r_embs_re - h_embs_im * r_embs_im
			h_multiply_r_im = h_embs_re * r_embs_im + h_embs_im * r_embs_re 
		
			h_multiply_r = torch.cat([h_multiply_r_re.unsqueeze(3), h_multiply_r_im.unsqueeze(3)], dim=3)
		
			scores = (h_multiply_r - t_embs).norm(dim=-1).norm(dim=-1)
		elif mode in ['link_prediction_h', 'head']:
			r_multiply_t_re = r_embs_re * t_embs_re + r_embs_im * t_embs_im
			r_multiply_t_im = r_embs_re * t_embs_im - r_embs_im * t_embs_re

			r_multiply_t = torch.cat([r_multiply_t_re.unsqueeze(3), r_multiply_t_im.unsqueeze(3)], dim=3)
			scores = (r_multiply_t - h_embs).norm(dim=-1).norm(dim=-1)

		return scores

	def forward_transe(self, positions, mode):
		batch_size = len(positions)
		device = self.lm_model_given.device

		h_idx = torch.LongTensor([positions[i]['head'][0] for i in range(batch_size)]).to(device)
		r_idx = torch.LongTensor([positions[i]['rel'][0]  for i in range(batch_size)]).to(device)
		t_idx = torch.LongTensor([positions[i]['tail'][0] for i in range(batch_size)]).to(device)

		h_emb_list = []
		r_emb_list = []
		t_emb_list = []
		for i in range(batch_size):
			h_emb_list.append(self.ent_embeddings_transe(h_idx[i]).unsqueeze(0))
			r_emb_list.append(self.rel_embeddings_transe(r_idx[i]).unsqueeze(0))
			t_emb_list.append(self.ent_embeddings_transe(t_idx[i]).unsqueeze(0))

		
		h_embs = torch.cat(h_emb_list, dim=0)
		r_embs = torch.cat(r_emb_list, dim=0)
		t_embs = torch.cat(t_emb_list, dim=0)

		can_ent_emb = self.ent_embeddings_transe(torch.LongTensor(list(i for i in range(self.n_ent))).to(device))

		if mode in ['link_prediction_h', 'head']:
			triple_score = self.score_triples_transe(can_ent_emb.expand(batch_size, can_ent_emb.shape[0], can_ent_emb.shape[1]), r_embs.unsqueeze(1), t_embs.unsqueeze(1))
		elif mode in ['link_prediction_t', 'tail']:
			triple_score = self.score_triples_transe(h_embs.unsqueeze(1), r_embs.unsqueeze(1), can_ent_emb.expand(batch_size, can_ent_emb.shape[0], can_ent_emb.shape[1]))

		return triple_score

	def forward_rotate(self, positions, mode):
		batch_size = len(positions)
		device = self.lm_model_given.device

		h_idx = torch.LongTensor([positions[i]['head'][0] for i in range(batch_size)]).to(device)
		r_idx = torch.LongTensor([positions[i]['rel'][0]  for i in range(batch_size)]).to(device)
		t_idx = torch.LongTensor([positions[i]['tail'][0] for i in range(batch_size)]).to(device)

		h_emb_list = []
		r_emb_list = []
		t_emb_list = []
		for i in range(batch_size):
			h_emb_list.append(self.ent_embeddings_transe(h_idx[i]).unsqueeze(0))
			r_emb_list.append(self.rel_embeddings_transe(r_idx[i]).unsqueeze(0))
			t_emb_list.append(self.ent_embeddings_transe(t_idx[i]).unsqueeze(0))

		
		h_embs = torch.cat(h_emb_list, dim=0)
		r_embs = torch.cat(r_emb_list, dim=0)
		t_embs = torch.cat(t_emb_list, dim=0)

		h_embs = h_embs.resize(batch_size, h_embs.shape[1]//2, 2)
		r_embs = r_embs.resize(batch_size, r_embs.shape[1]//2, 2)
		t_embs = t_embs.resize(batch_size, t_embs.shape[1]//2, 2)

		can_ent_emb = self.ent_embeddings_transe(torch.LongTensor(list(i for i in range(self.n_ent))).to(device))

		can_ent_emb = can_ent_emb.resize(self.n_ent, can_ent_emb.shape[1]//2, 2)
	

		if mode in ['link_prediction_h', 'head']:
			triple_score = self.score_triples_rotate(can_ent_emb.expand(batch_size, can_ent_emb.shape[0], can_ent_emb.shape[1], can_ent_emb.shape[2]), \
			 	r_embs.unsqueeze(1), t_embs.unsqueeze(1), mode)
		elif mode in ['link_prediction_t', 'tail']:
			triple_score = self.score_triples_rotate(h_embs.unsqueeze(1), r_embs.unsqueeze(1), \
				can_ent_emb.expand(batch_size, can_ent_emb.shape[0], can_ent_emb.shape[1], can_ent_emb.shape[2]), mode)

		return triple_score

	def encode_target(self, inputs, positions, mode):

		batch_size = len(positions)
		device = self.lm_model_target.device
		
		target_idx = torch.LongTensor([positions[i][0] for i in range(batch_size)]).to(device)
		target_pos = torch.LongTensor([positions[i][1] for i in range(batch_size)]).to(device)

		if not self.add_tokens:
			input_ids = inputs.pop('input_ids')
			input_embeds = self.lm_model_given.embeddings.word_embeddings(input_ids).squeeze(1)

			for i in range(batch_size):
				if mode != 'link_prediction_r':
					input_embeds[i, target_pos[i], :] = self.ent_embeddings(target_idx[i])
				else:
					input_embeds[i, target_pos[i], :] = self.rel_embeddings(target_idx[i])

			inputs['inputs_embeds'] = input_embeds

		logits = self.lm_model_target(**inputs) 

		target_embs = logits[0][:, 1, :]

		return target_embs

	def match(self, target_preds, target_encoded, triple_degrees, mode):
		device = self.lm_model_given.device

		sim = torch.zeros(target_preds.shape[0], target_encoded.shape[0]).to(self.lm_model_given.device)
		for it, target_pred in enumerate(target_preds):
			triple_degree = triple_degrees[it]
			h_deg, r_deg, t_deg = torch.tensor(triple_degree).float().to(device) 
			h_deg, r_deg, t_deg = h_deg.unsqueeze(0), r_deg.unsqueeze(0), t_deg.unsqueeze(0)
			h_logdeg, r_logdeg, t_logdeg = (h_deg+1).log(), (r_deg+1).log(), (t_deg+1).log()

			
			if mode == 'link_prediction_h': 
				deg_feature = torch.cat([h_logdeg, t_logdeg], dim=-1)
			elif mode == 'link_prediction_t':
				deg_feature = torch.cat([t_logdeg, h_logdeg], dim=-1)

			target_pred = target_pred.expand(target_encoded.shape[0], target_pred.shape[0])
			deg_feature = deg_feature.expand(target_encoded.shape[0], deg_feature.shape[0])
			#sim[it] = self.sim_classifier(torch.cat([target_pred, target_encoded, target_pred - target_encoded, target_pred * target_encoded], dim=-1)).T
			sim[it] = self.sim_classifier(torch.cat([target_pred, target_encoded, target_pred - target_encoded, target_pred * target_encoded, deg_feature], dim=-1)).T
		return sim
