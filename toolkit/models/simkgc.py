from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from transformers.file_utils import ModelOutput as TransformerModelOutput

# from triplet_mask import construct_mask
from .utils import construct_mask

__all__ = {
    "SimKGCModel",
    "SimKGCPredictor"
}
# def build_model(args) -> nn.Module:
    # return CustomBertModel(args)

# class SimKGCPredictor:

#     def __init__(self):
#         self.model = None
#         self.train_args = AttrDict()
#         self.use_cuda = False

#     def load(self, ckt_path, use_data_parallel=False):
#         assert os.path.exists(ckt_path)
#         ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
#         self.train_args.__dict__ = ckt_dict['args']
#         self._setup_args()
#         build_tokenizer(self.train_args)
#         self.model = build_model(self.train_args)

#         # DataParallel will introduce 'module.' prefix
#         state_dict = ckt_dict['state_dict']
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 k = k[len('module.'):]
#             new_state_dict[k] = v
#         self.model.load_state_dict(new_state_dict, strict=True)
#         self.model.eval()

#         if use_data_parallel and torch.cuda.device_count() > 1:
#             logger.info('Use data parallel predictor')
#             self.model = torch.nn.DataParallel(self.model).cuda()
#             self.use_cuda = True
#         elif torch.cuda.is_available():
#             self.model.cuda()
#             self.use_cuda = True
#         logger.info('Load model from {} successfully'.format(ckt_path))

#     def _setup_args(self):
#         for k, v in args.__dict__.items():
#             if k not in self.train_args.__dict__:
#                 logger.info('Set default attribute: {}={}'.format(k, v))
#                 self.train_args.__dict__[k] = v
#         logger.info('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
#         args.use_link_graph = self.train_args.use_link_graph
#         args.is_test = True

#     @torch.no_grad()
#     def predict_by_examples(self, examples: List[Example]):
#         data_loader = torch.utils.data.DataLoader(
#             Dataset(path='', examples=examples, task=args.task),
#             num_workers=1,
#             batch_size=max(args.batch_size, 512),
#             collate_fn=collate,
#             shuffle=False)

#         hr_tensor_list, tail_tensor_list = [], []
#         for idx, batch_dict in enumerate(data_loader):
#             if self.use_cuda:
#                 batch_dict = move_to_cuda(batch_dict)
#             outputs = self.model(**batch_dict)
#             hr_tensor_list.append(outputs['hr_vector'])
#             tail_tensor_list.append(outputs['tail_vector'])

#         return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

#     @torch.no_grad()
#     def predict_by_entities(self, entity_exs) -> torch.tensor:
#         examples = []
#         for entity_ex in entity_exs:
#             examples.append(Example(head_id='', relation='',
#                                     tail_id=entity_ex.entity_id))
#         data_loader = torch.utils.data.DataLoader(
#             Dataset(path='', examples=examples, task=args.task),
#             num_workers=2,
#             batch_size=max(args.batch_size, 1024),
#             collate_fn=collate,
#             shuffle=False)

#         ent_tensor_list = []
#         for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
#             batch_dict['only_ent_embedding'] = True
#             if self.use_cuda:
#                 batch_dict = move_to_cuda(batch_dict)
#             outputs = self.model(**batch_dict)
#             ent_tensor_list.append(outputs['ent_vectors'])

#         return torch.cat(ent_tensor_list, dim=0)

class ModelOutput(TransformerModelOutput):
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class SimKGCModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.model_name_or_path)
        self.tail_bert = deepcopy(self.hr_bert)

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        """
        input with (h, r ,t) , get the hr embedding and tail embedding.
        
        """
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return ModelOutput({'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()})

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        # assert tail_vector.size(0) == self.batch_size
        self.batch_size = len(batch_dict['batch_data'])
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t().float())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            # 取反
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        try:
            self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        except:
            import IPython; IPython.embed(); exit(1)
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)


        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, input_ids, attention_mask, token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=input_ids,
                                   mask=attention_mask,
                                   token_type_ids=token_type_ids)
        return ent_vectors.detach()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pooling", type=str, default="mean", help="The way of pooling after BERT encoder.")
        parser.add_argument("--t", type=float, default=0.05, help="ttt")
        parser.add_argument("--finetune_t", type=bool, default=True, help="ttt")
        parser.add_argument("--pre-batch", type=int, default=2, help="ttt")
        parser.add_argument("--pre-batch-weight", type=float, default=0.5, help="ttt")
        parser.add_argument("--use_self_negative", type=int, default=0, help="ttt")
        parser.add_argument('--additive-margin', default=0.02, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
        return parser

def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector