# Required environment variables:
# batch_size: (recommendation: 16 / 8)
# knn_topk: (recommendation: 64)
# knn_lambda: (recommendation: 0.2 /0.01)
# kshot: 16-1, 16-2 or 16-3

kshot=16-1
knn_topk=64
knn_lambda=0.2

CUDA_VISIBLE_DEVICES=0 python main.py --num_workers=8 \
    --model_name_or_path roberta-large \
    --batch_size 16 \
    --data_dir dataset/tacrev/kshot/$kshot \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --litmodel_class CombineEntityEmbeddingLitModel \
    --task_name wiki80 \
    --use_template_words 0 \
    --init_type_words 0 \
    --knn_lambda ${knn_lambda} \
    --knn_topk ${knn_topk} \
    --best_model your_model_ckpt
