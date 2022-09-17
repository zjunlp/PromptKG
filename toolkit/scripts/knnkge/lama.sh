CUDA_VISIBLE_DEVICES=1 python eval.py  --num_workers=16 \
   --model_name_or_path  bert-base-uncased \
   --model_class BertForMaskedLM \
   --lit_model_class LAMALitModel \
   --data_class LAMADataModule \
   --batch_size 256 \
   --dataset LAMA \
   --checkpoint output/model.ckpt \
   --eval_batch_size 128 \