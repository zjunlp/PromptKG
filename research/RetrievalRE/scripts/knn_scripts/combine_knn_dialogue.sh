
knn_topk=64
knn_lambda=0.2

CUDA_VISIBLE_DEVICES=0 python main.py --num_workers=8 \
    --model_name_or_path roberta-base \
    --batch_size 16 \
    --data_dir dataset/dialogue \
    --data_class DIALOGUE \
    --max_seq_length 512 \
    --model_class RobertaForPrompt \
    --litmodel_class CombineEntityEmbeddingLitModelDialogue \
    --task_name normal \
    --knn_lambda $knn_lambda \
    --knn_topk $knn_topk \
    --best_model your_model_ckpt
