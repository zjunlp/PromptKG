TAG=1
TYPE="prompt"
BS=8
LR=1e-5
MODEL="data/model_data/roberta_large"
MODEL_NAME="roberta_large"

K=16
MAX_STEP=800
EVAL_STEP=80
# MAX_LENGTH=128

REAL_BS=4
GS=$(expr $BS / $REAL_BS)
PROMPT_LR=1e-5
VIRTUAL_DEMO_INIT="random"
LAMBDA_CL=1.0


for TASK in "SST-2"
do

TASK_EXTRA=""
case $TASK in
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        DEMO_TEMPLATE=*sent_0*_It_was*mask*.
        MAPPING="{'0':'terrible','1':'great'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*sent_1**sep+*
        DEMO_TEMPLATE=*sent_0**mask*,*sent_1*
        MAPPING="{'0':'No','1':'Yes'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*sent_1**sep+*
        DEMO_TEMPLATE=*sent_0**mask*,*sent_1*
        MAPPING="{'0':'No','1':'Yes'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    MNLI)
        TEMPLATE=*cls**sent_0*?*mask*,*sent_1**sep+*
        DEMO_TEMPLATE=*sent_0*?*mask*,*sent_1*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        MAX_LENGTH=220
        DEMO_MAX_LENGTH=220
        TASK_EXTRA="--first_sent_limit 220 --demo_first_sent_limit 100"
        ;;
    SNLI)
        TEMPLATE=*cls**sent_0*?*mask*,*sent_1**sep+*
        DEMO_TEMPLATE=*sent_0*?*mask*,*sent_1*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    QNLI)
        TEMPLATE=*cls**sent_0*?*mask*,*sent_1**sep+*
        DEMO_TEMPLATE=*sent_0*?*mask*,*sent_1*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=256
        ;;
    RTE)
        TEMPLATE=*cls**sent_0*?*mask*,*sent_1**sep+*
        DEMO_TEMPLATE=*sent_0*?*mask*,*sent_1*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        MAX_LENGTH=220
        DEMO_MAX_LENGTH=220
        TASK_EXTRA="--first_sent_limit 200 --demo_first_sent_limit 100"
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        DEMO_TEMPLATE=*sent_0*_It_was*mask*.
        MAPPING="{0:'terrible',1:'great'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    sst-5)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        DEMO_TEMPLATE=*sent_0*_It_was*mask*.
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        DEMO_TEMPLATE=*sent_0*_This_is*mask*.
        MAPPING="{0:'subjective',1:'objective'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    trec)
        TEMPLATE=*cls**mask*:*sent_0**sep+*
        DEMO_TEMPLATE=*mask*:*sent_0**sep+*
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        DEMO_TEMPLATE=*sent_0*_It_was*mask*.
        MAPPING="{0:'terrible',1:'great'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
    mpqa)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        DEMO_TEMPLATE=*sent_0*_It_was*mask*.
        MAPPING="{0:'terrible',1:'great'}"
        MAX_LENGTH=128
        DEMO_MAX_LENGTH=128
        ;;
esac


VIRTUAL_DEMO=true
DEMO=true

for VIRTUAL_DEMO_LENGTH_PER_LABEL in 2
do
for SEED in 13 21 42 87 100
do
DATA_DIR=data/training_data/k_shot/$TASK/$K-$SEED
OUTPUT_DIR=data/output_data/$TASK-$TYPE-$K-$SEED-$MODEL_NAME

CUDA_VISIBLE_DEVICES=2 python run.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --prompt true \
  --virtual_demo $VIRTUAL_DEMO \
  --virtual_demo_init $VIRTUAL_DEMO_INIT \
  --virtual_demo_length_per_label $VIRTUAL_DEMO_LENGTH_PER_LABEL \
  --demo $DEMO \
  --demo_max_length $DEMO_MAX_LENGTH \
  --demo_template $DEMO_TEMPLATE \
  --lambda_cl $LAMBDA_CL \
  --template $TEMPLATE \
  --mapping $MAPPING \
  --model_name_or_path $MODEL \
  --few_shot_type $TYPE \
  --num_k $K \
  --max_seq_length $MAX_LENGTH \
  --per_device_train_batch_size $REAL_BS \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $EVAL_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 0 \
  --output_dir $OUTPUT_DIR \
  --seed $SEED \
  --tag $TAG \
  $TASK_EXTRA \

rm $OUTPUT_DIR/pytorch_model.bin $OUTPUT_DIR/vocab.json $OUTPUT_DIR/config.json $OUTPUT_DIR/merges.txt $OUTPUT_DIR/special_tokens_map.json

done
done
done