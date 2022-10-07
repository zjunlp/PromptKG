dataset="WN18RR"

CUDA_VISIBLE_DEVICES=0 python main.py  \
   --max_epochs=20  --num_workers=8 \
   --model_name_or_path  facebook/bart-base \
   --limit_val_batches 100 \
   --model_class BartKGC \
   --strategy="deepspeed_stage_2" \
   --lit_model_class KGBartLitModel \
   --label_smoothing 0.1 \
   --data_class KGT5DataModule \
   --precision 16 \
   --batch_size 64 \
   --check_val_every_n_epoch 5 \
   --wandb \
   --use_ce_loss 1 \
   --dataset ${dataset} \
   --eval_batch_size 50 \
   --beam_size 50 \
   --max_seq_length 128 \
   --lr 1e-4 


