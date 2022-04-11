kshot=16-1

CUDA_VISIBLE_DEVICES=0 python main.py  --num_workers=8 \
    --model_name_or_path roberta-large \
    --batch_size 16 \
    --data_dir dataset/semeval/k-shot/${kshot} \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --litmodel_class GetEntityEmbeddingLitModel \
    --task_name wiki80 \
    --use_template_words 0 \
    --init_type_words 0 \
    --best_model your_model_ckpt


