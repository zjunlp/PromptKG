subdataset="Google_RE"
# subdataset="Squad"
# subdataset="TREx"
# subdataset="ConceptNet"

CUDA_VISIBLE_DEVICES=2 python eval.py  --num_workers=16 \
   --model_name_or_path roberta-base \
   --model_class RobertaForMaskedLM \
   --lit_model_class LAMALitModel \
   --data_class LAMADataModule \
   --batch_size 256 \
   --dataset LAMA \
   --lamadataset ${subdataset}\
   --eval_batch_size 64 \