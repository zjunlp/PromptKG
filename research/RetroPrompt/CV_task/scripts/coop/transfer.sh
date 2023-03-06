#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=CoOp

CTP=end
NCTX=16
SHOTS=16
CSC=False

for SRC_DATASET in imagenet
do
    for TARGET_DATASET in imagenetv2 imagenet_r imagenet_a imagenet_sketch
    do
        for SHOTS in 16
        do
            if [ ${SHOTS} -eq 1 ]; then
                CFG=rn50_ep50
                EPOCH=50
            elif
                [ ${SHOTS} -eq 2 -o ${SHOTS} -eq 4 ]; then
                CFG=rn50_ep100
                EPOCH=100
            else
                CFG=rn50
                EPOCH=200
            fi

            for SEED in 1 2 3
            do
                MODEL_DIR=output/${SRC_DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${TARGET_DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir output/transfer/${TRAINER}/src_${SRC_DATASET}/target_${TARGET_DATASET}/${CFG}_${SHOTS}shots/seed${SEED} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${EPOCH} \
                --eval-only \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS}
            done
        done
    done
done