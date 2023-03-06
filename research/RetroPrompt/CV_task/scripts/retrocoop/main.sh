#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=RetroCoOp

CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
BETA=0.4

for DATASET in imagenet
do
    for TWK in True
    do
        for SHOTS in 1 2 4 8 16
        do
            for SEED in 1 2 3
            do
                if [ ${SHOTS} -eq 1 ]; then
                    if [ ${DATASET} == imagenet ]; then
                        CFG=rn50_ep25
                    else
                        CFG=rn50_ep50
                    fi
                elif
                    [ ${SHOTS} -eq 2 -o ${SHOTS} -eq 4 ]; then
                    if [ ${DATASET} == imagenet ]; then
                        CFG=rn50_ep50
                    else
                        CFG=rn50_ep100
                    fi
                else
                    if [ ${DATASET} == imagenet ]; then
                        CFG=rn50_ep100
                    else
                        CFG=rn50
                    fi
                fi

                OUTPUT_DIR=output/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/knn_train_${TWK}/beta_${BETA}/seed${SEED}
                CACHE_DIR=cache/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/knn_train_${TWK}/beta_${BETA}/seed${SEED}
                if [ -d "$DIR" ]; then
                    echo "Oops! The results exist at ${DIR} (so skip this job)"
                else
                    python train.py \
                    --root ${DATA} \
                    --seed ${SEED} \
                    --trainer ${TRAINER} \
                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                    --output-dir ${OUTPUT_DIR} \
                    --cache-dir ${CACHE_DIR} \
                    TRAINER.RETROCOOP.N_CTX ${NCTX} \
                    TRAINER.RETROCOOP.CSC ${CSC} \
                    TRAINER.RETROCOOP.CLASS_TOKEN_POSITION ${CTP} \
                    RETRIEVE.train_with_knn ${TWK} \
                    RETRIEVE.beta ${BETA} \
                    DATASET.NUM_SHOTS ${SHOTS}
                fi
            done
        done
    done
done