#/bin/bash

for EDIT_NUM in 2
do
CUDA_VISIBLE_DEVICES=2 python models/KE/run.py \
    --model_checkpoint checkpoints/PT_KGE_A-FB15k237 \
    --lr 5e-4 \
    --gpus 1 \
    --max_seq_length 64 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size ${EDIT_NUM} \
    --edit_num ${EDIT_NUM} \
    --stable_batch_size 16 \
    --max_steps 2000 \
    --divergences kl \
    --task_name add \
    --data_dir datasets/FB15k237/AddKnowledge \
    --data_type FB15k237 \
    --dirpath models/KE/logger/A-FB15k237
done
