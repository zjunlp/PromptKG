dataset="WN18RR"

CUDA_VISIBLE_DEVICES=2 python main.py  --max_epochs=10  --num_workers=16 \
   --model_name_or_path  bert-base-uncased \
   --num_sanity_val_steps 0 \
   --model_class KGBERTModel \
   --lit_model_class KGBERTLitModel \
   --data_class KGBERTDataModule \
   --label_smoothing 0.1 \
   --batch_size 32 \
   --check_val_every_n_epoch 1 \
   --wandb \
   --dataset ${dataset} \
   --eval_batch_size 64 \
   --max_seq_length 256 \
   --lr 3e-4 