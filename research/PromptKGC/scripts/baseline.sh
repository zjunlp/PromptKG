CUDA_VISIBLE_DEVICES=1 python main.py --gpus "0," --max_epochs=100  --num_workers=16 \
   --model_name_or_path  bert-base-cased \
   --accumulate_grad_batches 1 \
   --model_class KGBERT \
   --batch_size 32 \
   --check_val_every_n_epoch 1 \
   --data_dir dataset/WN18RR/42 \
   --max_seq_length 70 \
   --lr 5e-5 \
   --eval_batch_size 5000