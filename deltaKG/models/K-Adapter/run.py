# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Pre-train Factual Adapter
"""
import torch
import random
import numpy as np
import sys, os
import shutil
import warnings

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import logging
import os
from argparse import ArgumentParser
from pathlib import Path
import time

sys.path.append(os.getcwd())
from src.data.data_module import dataset_mapping
from src.models.models import PretrainedModel
from src.models.trainer import AdapterTrainer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = AdapterTrainer.add_argparse_args(parser)
    
    # args
    args = parser.parse_args()
    
    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]

    name_prefix = 'maxlen-' + str(args.max_seq_length) + '_' + 'batch-' + str(
        args.per_gpu_train_batch_size) + '_' + 'lr-' + str(args.learning_rate) + '_' + 'warmup-' + str(
        args.warmup_steps) + '_' + 'epoch-' + str(args.num_train_epochs) + '_' + str(args.comment)
    args.my_model_name = args.task_name + '_' + name_prefix
    args.output_dir = os.path.join(args.output_dir, args.task_name)

    if args.eval_steps is None:
        args.eval_steps = args.save_steps * 10

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    
    kgc_data = dataset_mapping[args.kge_model_type](args)
    pretrained_model = PretrainedModel(args.pretrain_model_checkpoint, args.kge_model_type)
    
    trainer = AdapterTrainer(args, pretrained_model, kgc_data)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        pass

    logger.info("Training/evaluation parameters %s", args)

    if args.do_eval:  
        logger.info("Validation sanity check")  
        trainer.evaluate("Vanity Check")
    
    # Training
    if args.do_train:
        global_step, tr_loss = trainer.train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
    if args.do_eval:    
        trainer.evaluate("Eval")
