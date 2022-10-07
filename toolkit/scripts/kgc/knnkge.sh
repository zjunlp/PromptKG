dataset="WN18RR"

CUDA_VISIBLE_DEVICES=0 python main.py  --max_epochs=10  --num_workers=16 \
   --model_name_or_path  bert-base-uncased \
   --num_sanity_val_steps 0 \
   --model_class KNNKGEModel \
   --lit_model_class KNNKGEPretrainLitModel \
   --label_smoothing 0.1 \
   --data_class KNNKGEPretrainDataModule \
   --batch_size 32 \
   --check_val_every_n_epoch 1 \
   --precision 16 \
   --wandb \
   --dataset ${dataset} \
   --eval_batch_size 65 \
   --max_seq_length 256 \
   --max_entity_length 256 \
   --lr 3e-4 



CUDA_VISIBLE_DEVICES=0 python main.py  --max_epochs=10  --num_workers=16 \
   --model_name_or_path  output/${dataset}/knnkge_pretrain_model \
   --num_sanity_val_steps 0 \
   --strategy="deepspeed_stage_2" \
   --model_class KNNKGEModel \
   --lit_model_class KNNKGELitModel \
   --label_smoothing 0.1 \
   --wandb \
   --data_class KNNKGEDataModule \
   --precision 16 \
   --batch_size 128 \
   --check_val_every_n_epoch 1 \
   --dataset ${dataset} \
   --eval_batch_size 256 \
   --max_seq_length 128 \
   --max_entity_length 64 \
   --lr 5e-5 
