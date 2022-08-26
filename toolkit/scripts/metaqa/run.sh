
CUDA_VISIBLE_DEVICES=2 python main.py  --max_epochs=10  --num_workers=16 \
   --model_name_or_path  t5-base \
   --num_sanity_val_steps 0 \
   --model_class T5KBQAModel \
   --lit_model_class KGT5LitModel \
   --label_smoothing 0.1 \
   --data_class MetaQADataModule \
   --precision 16 \
   --batch_size 64 \
   --check_val_every_n_epoch 2 \
   --dataset metaQA \
   --k_hop 1 \
   --eval_batch_size 128 \
   --max_seq_length 64 \
   --max_entity_length 128 \
   --lr 5e-5 


