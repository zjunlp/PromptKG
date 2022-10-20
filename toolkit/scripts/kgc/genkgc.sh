dataset="FB15k-237"

CUDA_VISIBLE_DEVICES=1 python main.py  \
   --max_epochs=21  --num_workers=8 \
   --model_name_or_path  facebook/bart-base \
   --model_class BartKGC \
   --strategy="deepspeed_stage_2" \
   --lit_model_class KGBartLitModel \
   --label_smoothing 0.1 \
   --data_class KGT5DataModule \
   --precision 16 \
   --batch_size 64 \
   --accumulate_grad_batches 4 \
   --prefix_tree_decode 1 \
   --check_val_every_n_epoch 10 \
   --wandb \
   --use_ce_loss 1 \
   --dataset ${dataset} \
   --eval_batch_size 4 \
   --beam_size 10 \
   --max_seq_length 128 \
   --lr 1e-4 


