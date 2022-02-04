python main.py --gpus "1," --max_epochs=4  --num_workers=0 \
   --model_name_or_path  facebook/bart-base \
   --accumulate_grad_batches 1 \
   --model_class BartKGC \
   --batch_size 128 \
   --check_val_every_n_epoch 4 \
   --data_dir dataset/wikidata5m \
   --pretrain 1 \
   --precision 16 \
   --wandb \
   --use_label_type 0 \
   --num_sanity_val_steps 0 \
   --output_full_sentence 0 \
   --prefix_tree_decode 1 \
   --max_seq_length 64 \
   --lr 1e-4

