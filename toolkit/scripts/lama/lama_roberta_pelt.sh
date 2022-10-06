subdataset="Google_RE"
# subdataset="TREx"
CUDA_VISIBLE_DEVICES=1 python eval.py  --num_workers=16 \
   --model_name_or_path roberta-base \
   --model_class RobertaEntForMaskedLM \
   --lit_model_class LAMALitModel \
   --data_class LAMADataModule \
   --batch_size 256 \
   --dataset LAMA \
   --lamadataset ${subdataset}\
   --eval_batch_size 64 \
   --pelt 1 \