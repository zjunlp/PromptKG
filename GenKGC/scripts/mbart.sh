# english en_XX 
# Jpanese ja_XX
# Chinese zh_CN



python main.py --gpus "1," --max_epochs=20  --num_workers=32 \
   --model_name_or_path  facebook/mbart-large-cc25 \
   --accumulate_grad_batches 1 \
   --model_class mBartKGC \
   --batch_size 64 \
   --check_val_every_n_epoch 10 \
   --limit_val_batches 0.1 \
   --data_dir dataset/X-en \
   --preseqlen 10 \
   --overwrite_cache \
   --output_full_sentence 1 \
   --prefix_tree_decode 0 \
   --max_seq_length 80 \
   --src_lang en_XX \
   --tgt_lang en_XX \
   --lr 5e-5

