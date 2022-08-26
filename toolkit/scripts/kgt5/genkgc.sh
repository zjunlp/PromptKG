CUDA_VISIBLE_DEVICES=3 python main.py  \
   --max_epochs=40  --num_workers=8 \
   --model_name_or_path  facebook/bart-base \
   --num_sanity_val_steps 0 \
   --model_class BartKGC \
   --lit_model_class KGBartLitModel \
   --label_smoothing 0.1 \
   --data_class KGT5DataModule \
   --precision 16 \
   --batch_size 64 \
   --check_val_every_n_epoch 10 \
   --wandb \
   --use_ce_loss 1 \
   --dataset WN18RR \
   --eval_batch_size 24 \
   --max_seq_length 128 \
   --lr 1e-4 


