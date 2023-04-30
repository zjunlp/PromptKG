for EX_SIZE in 32
do
for LAYER in "10,11"
do
for LR in 3e-4
do
echo ${LR}
echo ${EX_SIZE}
echo ${LAYER}
CUDA_VISIBLE_DEVICES=0 python models/CaliNet/run.py \
        --model_name_or_path checkpoints/PT_KGE_A-WN18RR \
        --do_train \
        --do_eval \
        --evaluation_strategy steps \
        --eval_steps 2500 \
        --lr_scheduler_type constant \
        --adafactor True \
        --max_source_length 64 \
        --max_target_length 8 \
        --output_dir models/CaliNet/output/WN18RR/add \
        --per_device_train_batch_size=128 \
        --per_device_eval_batch_size=32 \
        --stable_batch_size 8 \
        --overwrite_output_dir \
        --predict_with_generate \
        --text_column src_sent \
        --learning_rate ${LR} \
        --seed 1 \
        --warmup_steps 100 \
        --summary_column tgt_sent \
        --gradient_accumulation_steps 1 \
        --save_strategy steps \
        --ex_size ${EX_SIZE} \
        --kb_layer ${LAYER} \
        --save_total_limit 1 \
        --logging_steps 50 \
        --save_steps 5000 \
        --save_total_limit 1 \
        --logging_strategy steps \
        --max_steps 15000 \
        --run_name test \
        --eval_accumulation_steps 1 \
        --disable_tqdm false \
        --task_name add \
        --save_model_name wn18rr_add \
        --data_dir datasets/WN18RR/AddKnowledge

done
done
done