CUDA_VISIBLE_DEVICES=0 python main.py --gpus "0," --max_epochs=30  --num_workers=16 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class KGBERT \
   --batch_size 32 \
   --check_val_every_n_epoch 1 \
   --data_dir dataset/WN18RR/666 \
   --max_seq_length 70 \
   --lr 3e-5 \
   --eval_batch_size 5000 