python main.py --gpus "1," --max_epochs 5  --num_workers 32 \
   --model_name_or_path  facebook/bart-base \
   --accumulate_grad_batches 1 \
   --model_class BartKGC \
   --batch_size 64 \
   --check_val_every_n_epoch 1 \
   --data_dir dataset/WN18RR \
   --eval_batch_size 1 \
   --precision 16 \
   --beam_size 50 \
   --wandb \
   --overwrite_cache \
   --use_demos 1 \
   --use_label_type 0 \
   --output_full_sentence 0 \
   --prefix_tree_decode 1 \
   --max_seq_length 64 \
   --lr 3e-5

