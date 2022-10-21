from transformers import BertForMaskedLM

class KNNKGEModel(BertForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser



import os
from dgl.contrib import KVClient, KVServer, read_ip_config
import socket
import torch
import math
from torch import nn
from argparse import ArgumentParser
from torch import distributed as dist
from pytorch_lightning import Callback
from torch.optim import Adam
import numpy as np

def is_overflow(n):
    norm = torch.norm(n)
    if norm == float('inf') or norm == float('nan'):
        return True
    return False

def row_sparse_adagrad(name, ID, data, target, args):
    """Row-Sparse Adagrad update function
    """
    lr = args['lr']
    original_name = name[0:-6]
    state_sum = target[original_name + '_state-data-']
    # import pdb; pdb.set_trace()
    grad_sum = (data * data).mean(1)
    state_sum.index_add_(0, ID, grad_sum)
    std = state_sum[ID]  # _sparse_mask
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    tmp = (-lr * data / std_values)
    target[name].index_add_(0, ID, tmp)


def sparse_adam(name, ID, data, target, args):
    lr = args['lr']
    b1, b2 = args['betas']
    eps = args['eps']

    original_name = name[0:-6]
    exp_avg_name = original_name + '_exp_avg-data-'
    exp_avg_sq_name = original_name + '_exp_avg_sq-data-'
    # Exponential moving average of gradient values
    exp_avg = target[exp_avg_name][ID]
    # Exponential moving average of squared gradient values
    exp_avg_sq = target[exp_avg_sq_name][ID]
    step = target[original_name + '_step-data-'][0].add_(1)

    exp_avg.mul_(b1).add_(1 - b1, data)
    exp_avg_sq.mul_(b2).addcmul_(1 - b2, data, data)

    # bias_correction1 = (1-torch.zeros_like(step).fill_(b1).pow_(step)).view(-1, 1)
    # bias_correction2 = (1-torch.zeros_like(step).fill_(b2).pow_(step)).view(-1, 1)
    # denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
    bias_correction1 = 1 - b1 ** step
    bias_correction2 = 1 - b2 ** step
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1
    update = -step_size * exp_avg / denom
    # print('grad: {:.3f}, update: {:.3f}, exp_avg: {:.3f}, exp_avg_sq: {:.3f}'.format(
    #     torch.norm(data), torch.norm(update), torch.norm(exp_avg), torch.norm(exp_avg_sq)))

    target[name].index_add_(0, ID, update)
    target[exp_avg_name][ID] = exp_avg
    target[exp_avg_sq_name][ID] = exp_avg_sq


class EmbServer(KVServer):
    def __init__(self, server_id, server_namebook, num_client, queue_size=20 * 1024 * 1024 * 1024, net_type='socket'):
        super().__init__(server_id, server_namebook, num_client, queue_size, net_type)
        self._udf_push_handler = sparse_adam
        self._args = {}

    def set_args(self, args):
        self._args.update(args)

    def set_push_handler(self, handler):
        self._udf_push_handler = handler

    def _push_handler(self, name, ID, data, target):
        """push gradient only"""
        self._udf_push_handler(name, ID, data, target, self._args)


class EmbClient(KVClient):
    def __init__(self, server_namebook, queue_size=20 * 1024 * 1024 * 1024, net_type='socket'):
        super().__init__(server_namebook, queue_size, net_type)
        self._udf_push_handler = sparse_adam
        self._args = {}

    def set_args(self, args):
        self._args.update(args)

    def set_push_handler(self, handler):
        self._udf_push_handler = handler

    def _push_handler(self, name, ID, data, target):
        """push gradient only"""
        self._udf_push_handler(name, ID, data, target, self._args)


def check_port_available(port):
    """Return True is port is available to use
    """
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('', port))  ## Try to open port
        except OSError as e:
            if e.errno == 98:  ## Errorno 98 means address already bound
                return False
            raise e
        s.close()

        return True


def start_server(args):
    # torch.set_num_threads(1)

    server_namebook = read_ip_config(filename=args.ip_config)

    port = server_namebook[args.server_id][2]
    if check_port_available(port) == False:
        print("Error: port %d is not available." % port)
        exit()

    my_server = EmbServer(server_id=args.server_id,
                          server_namebook=server_namebook,
                          num_client=args.total_client)

    my_server.set_args({'lr': args.lr, 'betas': (0.9, 0.999), 'eps': 1e-8})
    my_server.set_push_handler(sparse_adam)

    if os.path.exists(args.ent_emb):
        print('load pretrained entity embedding: {}.\nnum_ent and ent_dim are ignored.'.format(args.ent_emb))
        ent_emb = np.load(args.ent_emb)
        if args.add_special_tokens:
            add_embs = np.random.randn(3, ent_emb.shape[1]) # add <pad>, <mask>, <unk>
            ent_emb = np.r_[add_embs, ent_emb]
        entity_emb = torch.from_numpy(ent_emb).float()
        print('shape: ', entity_emb.shape)
    else:
        # word embedding init is randn?
        entity_emb = torch.randn((args.num_ent, args.ent_dim))
    name = args.emb_name
    print('starting server...')
    my_server.init_data(name=name, data_tensor=entity_emb)
    my_server.init_data(name=name + '_exp_avg', data_tensor=torch.zeros_like(entity_emb))
    my_server.init_data(name=name + '_exp_avg_sq', data_tensor=torch.zeros_like(entity_emb))
    my_server.init_data(name=name + '_step', data_tensor=torch.zeros(entity_emb.size(0)))
    # my_server.init_data(name=name + '_state', data_tensor=torch.zeros(entity_emb.size(0)))

    print('KVServer listen at {}:{}'.format(my_server._ip, my_server._port))
    my_server.start()


class LargeEmbedding(nn.Module):
    """
    :param ip_config: path to ip_config file
    :param emb_name: name of emb in server
    :param lr: learning rate
    :param num_emb: number of embeddings
    """

    def __init__(self, ip_config, emb_name, lr, num_emb):
        super().__init__()
        server_namebook = read_ip_config(ip_config)
        self.client = EmbClient(server_namebook)
        optim_args = {'lr': lr, 'betas': (0.9, 0.999), 'eps': 1e-8}
        self.client.set_args(optim_args)
        self.client.set_push_handler(sparse_adam)
        self.client.connect()
        # if not dist.is_initialized() or get_local_rank() == 0:
        self.client.set_partition_book(emb_name, torch.zeros(num_emb))
        # else:
        #     self.client.set_partition_book(emb_name, None)
        self.name = emb_name
        self.trace = []
        self.num_emb = num_emb

    def __del__(self):
        self.client.shut_down()

    def forward(self, idx):
        """pull emb from server"""
        with torch.no_grad():
            bsz, slen = idx.size()
            cpu_idx = idx.cpu()
            unique_idx = torch.unique(cpu_idx)
            unique_emb = self.client.pull(self.name, unique_idx)
            gpu_emb = unique_emb.to(idx.device).detach_().requires_grad_(True)
        idx_mapping = {i.item(): j for j, i in enumerate(unique_idx)}
        mapped_idx = torch.zeros((bsz, slen), dtype=torch.long)
        for i in range(bsz):
            for j in range(slen):
                mapped_idx[i][j] = idx_mapping[cpu_idx[i][j].item()]

        # emb = torch.index_select(gpu_emb, 0, mapped_idx.cuda().view(-1)).view(bsz, slen, -1)
        emb = torch.embedding(gpu_emb, mapped_idx.cuda())
        # print('emb norm: {:.3f}, dtype: {}'.format(torch.norm(emb.data), emb.dtype))
        if self.training:
            self.trace.append((unique_idx, gpu_emb))
        return emb

    def update(self, skip=False):
        """push grad to server"""
        # print('update skip: {}'.format(skip))
        if skip:
            self.trace.clear()
            return
        with torch.no_grad():
            for idx, gpu_emb in self.trace:
                if gpu_emb.grad is not None:
                    grad = gpu_emb.grad.cpu()
                    self.client.push(self.name, idx, grad)
            self.trace.clear()

    def save(self, save_path):
        """pull all entity embeddings from server and save to disk"""
        with torch.no_grad():
            ent_emb = self.client.pull(self.name, torch.arange(0, self.num_emb))
            ent_emb = ent_emb.cpu().detach_().numpy()
        np.save(os.path.join(save_path, 'entities.npy'), ent_emb)


class EmbUpdateCallback(Callback):
    def __init__(self, large_emb):
        super().__init__()
        self.large_emb = large_emb

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx: int):
        # gradient overflow时 optimizer就会被patch
        if hasattr(optimizer, '_amp_stash'):
            # print('optimizer patched {}'.format(self.optimizer._amp_stash.already_patched))
            if optimizer._amp_stash.already_patched:
                self.large_emb.update(skip=True)
                return
        self.large_emb.update(skip=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate to update the emb')
    parser.add_argument('--server_id', type=int, default=0, help='id of server, from 0 to (num of server - 1)')
    parser.add_argument('--ip_config', type=str, default='emb_ip.cfg',
                        help='path to config of server, every line in file is [ip] [port] [Num of server]')
    parser.add_argument('--emb_name', type=str, default='entity_emb',
                        help='name of the embedding, should match with clients\' side')
    parser.add_argument('--num_ent', type=int, default=30000, help='num of embeddings')  # 500w
    parser.add_argument('--ent_dim', type=int, default=200, help='embedding dim')
    parser.add_argument('--total_client', type=int, default=1, help='num of client')
    parser.add_argument('--add_special_tokens', action='store_true', help='whether to add special tokens')
    parser.add_argument('--ent_emb', type=str, default="")
    args = parser.parse_args()
    start_server(args)