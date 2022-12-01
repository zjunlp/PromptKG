dataset="WN18RR"

python main.py  --max_epochs=10  --num_workers=8 \
   --model_name_or_path  bert-base-uncased \
   --num_sanity_val_steps 0 \
   --model_class SimKGCModel \
   --strategy="deepspeed_stage_2" \
   --lit_model_class SimKGCLitModel \
   --data_class SimKGCDataModule \
   --batch_size 96 \
   --precision 16 \
   --check_val_every_n_epoch 1 \
   --dataset ${dataset} \
   --eval_batch_size 256 \
   --max_entity_length 64 \
   --max_seq_length 150 \
   --lr 3e-5 


