dataset="WN18RR"

CUDA_VISIBLE_DEVICES=2 python main.py  \
   --max_epochs=40  --num_workers=8 \
   --model_name_or_path  t5-small \
   --num_sanity_val_steps 4 \
   --limit_val_batches 100 \
   --model_class T5KGC \
   --lit_model_class KGT5KGCLitModel \
   --label_smoothing 0.1 \
   --data_class KGT5DataModule \
   --batch_size 96 \
   --check_val_every_n_epoch 10 \
   --prefix_tree_decode 1 \
   --wandb \
   --dataset WN18RR \
   --overwrite_cache \
   --eval_batch_size 48 \
   --beam_size 10 \
   --max_seq_length 128 \
   --lr 1e-4 


