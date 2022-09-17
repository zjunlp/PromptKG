CUDA_VISIBLE_DEVICES=1,2 python main.py  --max_epochs=10  --num_workers=16 \
   --model_name_or_path  bert-base-uncased \
   --num_sanity_val_steps 0 \
   --strategy="deepspeed_stage_2" \
   --model_class SimKGCModel \
   --lit_model_class SimKGCLitModel \
   --data_class SimKGCDataModule \
   --precision 16 \
   --batch_size 128 \
   --check_val_every_n_epoch 1 \
   --dataset FB15k-237 \
   --eval_batch_size 256 \
   --max_seq_length 128 \
   --max_entity_length 56 \
   --lr 1e-5 


