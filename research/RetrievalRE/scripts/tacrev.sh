# Required environment variables:
# accumulate_grad_batches (recommendation: 4 / 8)
# lr: learning rate (recommendation: 3e-5 8e-5 5e-5)
# batch_size: (recommendation: 16 / 8)
# kshot: 16-1, 16-2 or 16-3

kshot=16-1

CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=10  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --data_dir dataset/tacrev/k-shot/$kshot \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --use_template_words 0 \
    --init_type_words 0 \
    --output_dir output/tacrev/kshot/$kshot