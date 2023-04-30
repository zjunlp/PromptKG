# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Pre-train fac-adapter

GPU='2'
TASK_NAME='edit'
CUDA_VISIBLE_DEVICES=$GPU python models/K-Adapter/run.py  \
        --model_type bert \
        --pretrain_model_checkpoint checkpoints/PT_KGE_E-FB15k237 \
        --model_name_or_path bert-base-uncased  \
        --data_dir datasets/FB15k237/EditKnowledge  \
        --output_dir models/K-Adapter/output/FB15k237 \
        --restore '' \
        --do_train  \
        --evaluate_during_training 'True' \
        --task_name $TASK_NAME     \
        --comment 'kgc-adapter' \
        --per_gpu_train_batch_size 32   \
        --per_gpu_eval_batch_size 32   \
        --num_train_epochs 100 \
        --max_seq_lengt 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --warmup_steps 1200 \
        --save_steps 1000 \
        --adapter_size 768 \
        --adapter_list "0,11" \
        --adapter_skip_layers 0 \
        --adapter_transformer_layers 2 \
        --meta_adapter_model=""