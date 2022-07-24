CUDA_VISIBLE_DEVICES= python main.py --gpus="0"  --max_epochs=25  --num_workers=16 \
   --accumulate_grad_batches 2 \
   --model_name_or_path  t5-small \
   --num_sanity_val_steps 0 \
   --model_class T5KGC \
   --batch_size 128 \
   --check_val_every_n_epoch 5 \
   --dataset FB15k-237 \
   --eval_batch_size 256 \
   --wandb \
   --prefix_tree_decode 1 \
   --max_seq_length 128 \
   --lr 3e-4 


