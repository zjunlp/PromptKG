#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=RetroCoOp

CTP=end
NCTX=16
CSC=False
BETA=0.4

for DATASET in imagenet
do
    for SHOTS in  1 2 4 8 16
    do
        if [ ${SHOTS} -eq 1 ]; then
            if [ ${DATASET} == imagenet ]; then
                CFG=rn50_ep25
                EPOCH=25
            else
                CFG=rn50_ep50
                EPOCH=50
            fi
        elif
            [ ${SHOTS} -eq 2 -o ${SHOTS} -eq 4 ]; then
            if [ ${DATASET} == imagenet ]; then
                CFG=rn50_ep50
                EPOCH=50
            else
                CFG=rn50_ep100
                EPOCH=100
            fi
        else
            if [ ${DATASET} == imagenet ]; then
                CFG=rn50_ep100
                EPOCH=100
            else
                CFG=rn50
                EPOCH=200
            fi
        fi

        for TWK in True False
        do
            for SEED in 1 2 3
            do
                MODEL_DIR=output/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/knn_train_${TWK}/beta_${BETA}/seed${SEED}
                CACHE_DIR=cache/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/knn_train_${TWK}/beta_${BETA}/seed${SEED}
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir output/evaluation/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/knn_train_${TWK}/beta_${BETA}/seed${SEED} \
                --cache-dir ${CACHE_DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${EPOCH} \
                --eval-only \
                TRAINER.RETROCOOP.N_CTX ${NCTX} \
                TRAINER.RETROCOOP.CSC ${CSC} \
                TRAINER.RETROCOOP.CLASS_TOKEN_POSITION ${CTP} \
                RETRIEVE.load_cache True \
                RETRIEVE.update_cache False \
                DATASET.NUM_SHOTS ${SHOTS}
            done
        done
    done
done