#!/bin/bash

DATASET=6_cardio

python train.py \
    --dataset ${DATASET} \
    --data_dir datasets/Classical \
    \
    --batch_size 64 \
    --num_workers 4 \
    \
    --flow_layers 16 \
    --mlp_layers 2 \
    --use_channel_wise_splits \
    --use_checkerboard_splits \
    --layer_norm \
    \
    --lr 1e-3 \
    \
    --output_dir exp/${DATASET} \
    --seed 42 \
    \
    --plot_histogram \