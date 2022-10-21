CUDA_VISIBLE_DEVICES=0 python main.py  --max_epochs=5  --num_workers=16 \
   --model_name_or_path  output/ml20m/kgrec_pretrain_model \
   --num_sanity_val_steps 0 \
   --model_class KGRECModel \
   --lit_model_class KGRECLitModel \
   --data_class KGRECDataModule \
   --check_val_every_n_epoch 2 \
   --batch_size 16 \
   --dataset ml20m \
   --eval_batch_size 64 \
   --max_seq_length 225 \
   --max_entity_length 128 \
   --early_stop 0 \
   --lr 5e-5 
   # --precision 16
   # --wandb \
   # --strategy="deepspeed_stage_2" \


