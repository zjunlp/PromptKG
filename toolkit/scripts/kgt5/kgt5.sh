python main.py  \
   --max_epochs=40  --num_workers=6 \
   --model_name_or_path  pretrained_models/t5-base \
   --num_sanity_val_steps 0 \
   --model_class T5KGC \
   --lit_model_class KGT5KGCLitModel \
   --label_smoothing 0.1 \
   --data_class KGT5DataModule \
   --precision 16 \
   --batch_size 64 \
   --check_val_every_n_epoch 10 \
   --dataset WN18RR \
   --eval_batch_size 24 \
   --max_seq_length 128 \
   --lr 1e-4 


