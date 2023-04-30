#/bin/bash

for EDIT_NUM in 2
do
CUDA_VISIBLE_DEVICES=3 python models/KE/run.py \
    --model_checkpoint checkpoints/PT_KGE_E-FB15k237 \
    --lr 5e-4 \
    --gpus 0, \
    --max_seq_length 64 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size ${EDIT_NUM} \
    --edit_num ${EDIT_NUM} \
    --stable_batch_size 16 \
    --max_steps 2000 \
    --divergences kl \
    --task_name edit \
    --data_dir datasets/FB15k237/EditKnowledge \
    --data_type FB15k237 \
    --dirpath models/KE/logger/E-FB15k237
done
