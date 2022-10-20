dataset="WN18RR"
model_name_or_path="bert-base-uncased"
devices=1

CUDA_VISIBLE_DEVICES=${devices} python main.py  --max_epochs=10  --num_workers=16 \
   --model_name_or_path  ${model_name_or_path} \
   --num_sanity_val_steps 0 \
   --model_class KNNKGEModel \
   --lit_model_class KNNKGEPretrainLitModel \
   --label_smoothing 0.1 \
   --data_class KNNKGEPretrainDataModule \
   --batch_size 16 \
   --check_val_every_n_epoch 1 \
   --wandb \
   --dataset ${dataset} \
   --eval_batch_size 64 \
   --max_seq_length 256 \
   --max_entity_length 256 \
   --lr 2e-5 



CUDA_VISIBLE_DEVICES=${devices} python main.py  --max_epochs=10  --num_workers=16 \
   --model_name_or_path  output/${dataset}/knnkge_pretrain_model \
   --num_sanity_val_steps 0 \
   --strategy="deepspeed_stage_2" \
   --model_class KNNKGEModel \
   --lit_model_class KNNKGELitModel \
   --label_smoothing 0.1 \
   --wandb \
   --data_class KNNKGEDataModule \
   --batch_size 16 \
   --accumulate_grad_batches 4 \
   --check_val_every_n_epoch 1 \
   --dataset ${dataset} \
   --eval_batch_size 16 \
   --max_seq_length 256 \
   --max_entity_length 64 \
   --lr 2e-5 
