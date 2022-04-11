# Required environment variables:
# accumulate_grad_batches (recommendation: 16)
# lr: learning rate (recommendation: 8e-5 5e-5)
# batch_size: (recommendation: 8 / 16)

CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=25  --num_workers=8 \
    --model_name_or_path roberta-base \
    --accumulate_grad_batches 16 \
    --batch_size 8 \
    --data_dir dataset/dialogue \
    --check_val_every_n_epoch 1 \
    --data_class DIALOGUE \
    --max_seq_length 512 \
    --model_class RobertaForPrompt \
    --litmodel_class DialogueLitModel \
    --task_name normal \
    --lr 8e-5 \
    --output_dir output/dialogue/full