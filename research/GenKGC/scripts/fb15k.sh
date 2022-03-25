python main.py --gpus "2," --max_epochs=15  --num_workers=32 \
   --model_name_or_path  facebook/bart-large \
   --accumulate_grad_batches 1 \
   --model_class BartKGC \
   --batch_size 64 \
   --check_val_every_n_epoch 5 \
   --data_dir dataset/FB15k-237 \
   --eval_batch_size 1 \
   --precision 16 \
   --wandb \
   --use_demos 0 \
   --use_label_type 0 \
   --output_full_sentence 0 \
   --prefix_tree_decode 1 \
   --max_seq_length 128 \
   --lr 5e-5

