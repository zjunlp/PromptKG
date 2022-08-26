CUDA_VISIBLE_DEVICES=3 python main.py --accelerator="gpu" --gpus="0," --devices=1 --max_epochs=10  --num_workers=16 \
   --accumulate_grad_batches 2 \
   --model_name_or_path  bert-base-uncased \
   --wandb \
   --num_sanity_val_steps 0 \
   --model_class SimKGCModel \
   --lit_model_class SimKGCLitModel \
   --data_class SimKGCDataModule \
   --batch_size 64 \
   --precision 16 \
   --check_val_every_n_epoch 1 \
   --dataset WN18RR \
   --eval_batch_size 256 \
   --max_entity_length 64 \
   --max_seq_length 150 \
   --lr 3e-5 


