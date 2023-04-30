MODEL_NAME="KGEditor"
DATASET_NAME="FB15k237"
TASK_NAME="edit"

while getopts ":m:d:t:" opt
do  
    case $opt in
        m)
        echo "the selected model is $OPTARG."
        MODEL_NAME=$OPTARG
        ;;
        d)
        echo "the selected dataset is $OPTARG."
        DATASET_NAME=$OPTARG
        ;;
        t)
        echo "the selected task is $OPTARG."
        TASK_NAME=$OPTARG
        ;;
        ?)
        echo "Unknown Param."
        exit 1;;
        :)
        echo "There is no param."
        ;;
esac done

echo "model_name: ${MODEL_NAME}; dataset_name: ${DATASET_NAME}; task_name: ${TASK_NAME}; "

bash scripts/${MODEL_NAME}/${MODEL_NAME}_${DATASET_NAME}_${TASK_NAME}.sh
