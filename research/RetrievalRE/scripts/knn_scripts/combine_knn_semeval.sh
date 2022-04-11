# Required environment variables:
# batch_size: (recommendation: 8)
# knn_topk: (recommendation: 64)
# knn_lambda: (recommendation: 0.5 / 0.2)
# kshot: 16-1, 16-2 or 16-3

kshot=16-1
knn_topk=64
knn_lambda=0.5

CUDA_VISIBLE_DEVICES=0 python main.py  --num_workers=8 \
    --model_name_or_path roberta-large \
    --batch_size 8 \
    --data_dir dataset/semeval/k-shot/$kshot \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --litmodel_class CombineEntityEmbeddingLitModel \
    --task_name wiki80 \
    --use_template_words 0 \
    --init_type_words 0 \
    --knn_topk $knn_topk \
    --knn_lambda $knn_lambda \
    --best_model your_model_ckpt
