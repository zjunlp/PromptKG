CUDA_VISIBLE_DEVICES=2 python main.py --gpus="0,"  --max_epochs=10  --num_workers=16 \
   --accumulate_grad_batches 2 \
   --model_name_or_path  bert-base-uncased \
   --num_sanity_val_steps 0 \
   --model_class SimKGCModel \
   --lit_model_class TransformerSimKGC \
   --data_class SimKGCDataModule \
   --batch_size 128 \
   --check_val_every_n_epoch 1 \
   --dataset WN18RR \
   --eval_batch_size 256 \
   --max_seq_length 128 \
   --lr 3e-5 


